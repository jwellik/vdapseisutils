from __future__ import (absolute_import, division, print_function)
import numpy as np


########################################################################################################################
### Basic Geometry
########################################################################################################################

def dd2dms(dd, hemisphere=None):
    """DD2DMS Convert decicmal degrees to degrees/minutes/seconds"""

    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd*3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -1*degrees

    if hemisphere == "latitude":
        hemi = "N" if degrees >= 0.0 else "S"
        degrees = abs(degrees)
        dms = (degrees, minutes, seconds, hemi)
    elif hemisphere == "longitude":
        hemi = "E" if degrees >= 0.0 else "W"
        degrees = abs(degrees)
        dms = (degrees, minutes, seconds, hemi)
    else:
        dms = (degrees, minutes, seconds)

    return dms


def dd2dm(dd, hemisphere=None):
    """DD2DM ConverT decicaml degrees to degrees/minutes"""

    degrees, minutes, seconds = dd2dms(dd)
    minutes = minutes + seconds/60.0
    del seconds

    if hemisphere == "latitude":
        hemi = "N" if degrees >= 0.0 else "S"
        degrees = abs(degrees)
        dm = (degrees, minutes, hemi)
    elif hemisphere == "longitude":
        hemi = "E" if degrees >= 0.0 else "W"
        degrees = abs(degrees)
        dm = (degrees, minutes, hemi)
    else:
        dm = (degrees, minutes)

    return dm


def dms2dd(dms):
    dd = np.abs(dms[0])+dms[1]/60+dms[2]/3600
    if dms[0] < 0:
        dd *= -1
    return dd


########################################################################################################################
### EXTENTS & BUFFERS
########################################################################################################################


def geodesic_point_buffer(lat, lon, km):
    '''https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python/289923'''
    ### Doesn't work near poles??? bc of azimuthal equidistant projection system???

    from functools import partial
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Point
    import numpy as np

    proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres

    a = np.array(transform(project, buf).exterior.coords[:])  # n-by-2 array of lon,lat pairs
    b = np.empty_like(a)
    b[:, 0] = a[:, 1]
    b[:, 1] = a[:, 0]
    # returns an n-by-2 array of lat,lon pairs
    return b


def radial_extent2map_extent(lat, lon, km):
    """Returns [minlon, maxlon, minlat, maxlat] LRBT"""
    rlatlon = geodesic_point_buffer(lat, lon, km)
    map_extent = [min(rlatlon[:, 1]), max(rlatlon[:, 1]), min(rlatlon[:, 0]), max(rlatlon[:, 0])]
    return map_extent


def radial_extent2bounds_bltr(lat, lon, km):
    """Returns [minlat, minlon, maxlat, maxlon] or [bottom, left, top, right]"""
    rlatlon = geodesic_point_buffer(lat, lon, km)
    map_extent = [min(rlatlon[:, 0]), min(rlatlon[:, 1]), max(rlatlon[:, 0]), max(rlatlon[:, 1])]
    return map_extent


def set_radial_map_extent(ax, lat, lon, km, crs=None):
    '''Sets the extent of the map based on radius from given point'''
    ax.set_extent(radial_extent2map_extent(lat, lon, km), crs=crs)
    return ax

########################################################################################################################
### SIGHTING POINTS
########################################################################################################################

def sight_point_geopy(origin, bearing, km):
    """Returns the (lat,lon) of point N km away along a given bearing"""
    # https://stackoverflow.com/questions/24427828/calculate-point-based-on-distance-and-direction
    import geopy.distance

    # Define starting point.
    start = geopy.Point(origin[0], origin[1])

    # Define a general distance object, initialized with a distance of 1 km.
    d = geopy.distance.VincentyDistance(kilometers=km)

    # Use the `destination` method with a bearing of 0 degrees (which is north)
    # in order to go from point `start` 1 km to north.
    return d.destination(point=start, bearing=bearing)


def sight_point_pyproj(origin, bearing, km, ellipse='WGS84'):
    """Returns the (lat,lon) of point N km away along a given bearing"""
    # https://gis.stackexchange.com/questions/174761/create-a-new-point-from-a-reference-point-degree-and-distance
    import pyproj

    endLon, endLat, backAzimuth = (pyproj.Geod(ellps=ellipse).fwd(origin[1], origin[0], bearing, km))
    point = (endLat, endLon)
    return point


def sight_point(origin, bearing, km, method="pyproj"):
    if method == "pyproj":
        return sight_point_pyproj(origin, bearing, km)
    elif method == "geopy":
        return sight_point_geopy(origin, bearing, km)
    else:
        print("sight_point: Method not understood :-(")


def backazimuth(*args, **kwargs):
    return backazimuth_pyproj(*args, **kwargs)


def backazimuth_pyproj(latlon1, latlon2, ellipse='WGS84'):
    # BACKAZIMUTH_PYPROJ Returns azimuth and distance (in meters) between two points

    from pyproj import Geod
    g = Geod(ellps=ellipse)  # Use Clarke 1866 ellipsoid.
    # specify the lat/lons of Boston and Portland.
    boston_lat = 42. + (15. / 60.);
    boston_lon = -71. - (7. / 60.)
    portland_lat = 45. + (31. / 60.);
    portland_lon = -123. - (41. / 60.)
    # az12, az21, dist = g.inv(boston_lon, boston_lat, portland_lon, portland_lat)  # example
    az12, az21, dist = g.inv(latlon1[1], latlon1[0], latlon2[1], latlon2[0])  # example
    return az12, dist


def project2line(lats, lons, P1=(-90, 0), P2=(90, 0), unit="m"):
    """
    PROJECT2LINE

    Project points to line. Line defined by two points.

    :param lats:
    :param lons:
    :param P1:
    :param P2:
    :return:
    """

    import pyproj
    import math

    # fwdA
    # dA    : EQ distance along A-A'
    # Create projection system
    geodesic = pyproj.Geod(ellps='WGS84')  # Create projection system
    # Angle of cross-section vectors
    fwdAA, backAA, distanceAA = geodesic.inv(P1[1], P1[0], P2[1], P2[0])  # (long, lat, long, lat)
    # Returns forward azimuth (degrees), backazimuth (degrees), and distance (meters)

    FWDAA = []
    BACKAA = []
    DISTANCEAA = []
    ALPHAAA = []
    DAA = []  # Distance from P1 to point along cross section

    # for idx, row in catdata.iterrows():
    for lat, lon in zip(lats, lons):
        # Get distance along cross section for each poi (point of interest)

        # azimuth and distance from P1 to poi (distanceA is distance in meters)
        fwdA, backA, distanceA = geodesic.inv(P1[1], P1[0], lon, lat)  # long, lat, long, lat

        # angle between A1-poi and A1-A2
        alphaA = fwdA - fwdAA  # angle between A1-pt and A1-A2

        # distance along cross-section (still in meters)
        dA = distanceA * math.cos((alphaA) * (np.pi / 180))  # distance to pt along xsection line

        FWDAA.append(fwdA)
        BACKAA.append(backA)
        DISTANCEAA.append(distanceA)
        ALPHAAA.append(alphaA)
        if unit == "m":
            DAA.append(dA)
        elif unit == "km":
            DAA.append(dA/1000)
        else:
            DAA.append(dA)
            print("Unit '{}' not understood. Options are 'm' or 'km'. Using 'm'.".format(unit))

    return DAA
