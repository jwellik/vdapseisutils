from __future__ import (absolute_import, division, print_function)

import pandas as pd
import numpy as np

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

import six

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
    """DD2DM Conver decicaml degrees to degrees/minutes"""

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


def backazimuth_pyproj(latlon1, latlon2, ellipse='WGS84'):
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


def project2line(lats, lons, P1=(-90, 0), P2=(90, 0)):
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

    FWDAA = []
    BACKAA = []
    DISTANCEAA = []
    ALPHAAA = []
    DAA = []  # Distance from P1 to point along cross section

    # for idx, row in catdata.iterrows():
    for lat, lon in zip(lats, lons):
        fwdA, backA, distanceA = geodesic.inv(P1[1], P1[0], lon, lat)  # long, lat, long, lat
        alphaA = fwdA - fwdAA  # angle between A1-pt and A1-A2
        dA = distanceA * math.cos((alphaA) * (np.pi / 180))  # distance to pt along xsection line
        FWDAA.append(fwdA)
        BACKAA.append(backA)
        DISTANCEAA.append(distanceA)
        ALPHAAA.append(alphaA)
        DAA.append(dA)

    return DAA


########################################################################################################################
### MAP FEATURES
########################################################################################################################

# def add_north_arrow()

# def add_scale_bar()

# def add_radius()


def location_map(fig, lat, lon, marker='or', location='top_left'):
    if location == 'top_right':
        position = [0.785, 0.785, 0.2, 0.2]
    elif location == 'top_left':
        position = [0, 0.785, 0.2, 0.2]
    else:
        position = [0.785, 0.785, 0.2, 0.2]
    ax = fig
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    ax.plot(lon, lat, marker)
    ax.set_global()
    ax.gridlines()
    return fig

########################################################################################################################
### IO
########################################################################################################################


def save(fig, filepath, format=['.png', '.svg']):
    for ext in ['.png', '.svg']:
        print('Saving {}'.format(ext))
        if ext == '.svg':
            pad_inches = 0
        else:
            pad_inches = 4
        fig.savefig(filepath + ext, pad_inches=pad_inches)
        # fig.savefig(outputfile+ext, dpi=grl.dpi, facecolor=None, edgecolor=None,
        #    orientation='portrait', papertype=None,
        #    transparent=True, bbox_inches=None, pad_inches=pad_inches,
        #    frameon=None, metadata=None)


########################################################################################################################
### EQ LOCATIONS
########################################################################################################################

# Use these better?
eqkwargs = dict(marker='o', markerfacecolor='black', markersize=8, alpha=0.95)
eqerrorargs = dict(color='black', alpha=0.95)

# Reads a csv file and puts it into an ObsPy Catalog object
# Transfers field names to object properties


def csv2catalog():
    pass


# Change this to create a dictionary with lat, lon, count, grid_size where lat, lon, count must be same size lists
def eqhypo2heatmap(catalogdf, grid_size):
    '''Converts EQhypocenters to a gridded heatmap
    INPUT
    catalog : Pandas DataFrame
        'lat' : latitude
        'lon' : longitude
    grid_size : float : degree spacing for the grid

    OUTPUT
    heatmap : Pandas DataFrame
        'lat'
        'lon'
        'count' : number of earthquakes in this bin
    '''
    # Convert catalog to heatmap based on gridded location by degree
    catalogdf = catalogdf.round(int(np.log10(grid_size) * -1))  # round to n decimals where n = np.log10(grid_size)*-1
    heatmap = catalogdf.groupby(['lat', 'lon']).agg('size')
    # Save heatmap in dataframe
    heatmap = pd.DataFrame({
        'lat': heatmap.index.get_level_values(0),
        'lon': heatmap.index.get_level_values(1),
        'count': heatmap.values
    })
    return heatmap


def catalog2heatmap(catalog, grid_size):
    pass


# More Heatmap references for future development:
# https://james-brennan.github.io/posts/fast_gridding_geopandas/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
# https://stackoverflow.com/questions/62778939/python-fastest-way-to-map-continuous-coordinates-to-discrete-grid


# def plot_heatmap(ax, catalog, grid_size, colormap='RdPu', ncmap=15):
def plot_heatmap(ax, catalog, grid_size, colormap='plasma', ncmap=15):
    heatmap = eqhypo2heatmap(catalog, grid_size)

    ######## Create Heat Map #########################################
    lon = heatmap['lon'].values
    lat = heatmap['lat'].values
    c = heatmap['count'].values

    from matplotlib import cm
    low = min(heatmap['count'])
    high = max(heatmap['count'])
    nrange = high - low + 1
    cmap = cm.get_cmap(colormap, ncmap)
    colors = []
    for c in heatmap['count'].values:
        i = int((c - low) / nrange * ncmap)
        colors.append(cmap(i))

    for x, y, c in zip(lon, lat, colors):
        heat_map_final = ax.fill(
            [x - grid_size / 2, x - grid_size / 2, x + grid_size / 2, x + grid_size / 2, x - grid_size / 2],
            [y - grid_size / 2, y + grid_size / 2, y + grid_size / 2, y - grid_size / 2, y - grid_size / 2],
            color=c[0:3], alpha=0.85, linewidth=0,
            transform=ccrs.PlateCarree())

    return ax


def plot_hypo(ax, lat, lon, transform=ccrs.Geodetic(), marker='o', color='black', markersize=8, alpha=0.95):
    ax.plot(lon, lat, color=color)
    return ax


def plot_catalog(axm, catalog, transform=ccrs.Geodetic(), plot_errors=True, verbose=False, **eqkwargs):
    '''Prints Hypocenters for ObsPy Catalog object
    Currently prints just the last origin available (assumes that's the preferred origin)

    **EQKWARGS Anything undertsood by matplotlib.pyplot.plot()
    '''

    if verbose: print('Printing {} event(s) from Catalog object.'.format(len(catalog)))
    for event in catalog:
        print('- Printing catalog event...')
        lat = event.origins[-1].latitude
        lon = event.origins[-1].longitude
        depth = event.origins[-1].depth

        # plot errors on bottom
        if plot_errors:
            axm = plot_eventerror2map(axm, event, **eqerrorargs, transform=transform)

        # Plot hypocenter on top
        axm = plot_hypo(axm, lat, lon, transform=transform, **eqkwargs)

    return axm


def plot_catalog2xs_dep(fig, catalog, marker='o', color='black', markersize=8, alpha=0.95,
                    plot_errors=True):
    # Get origin info for xsection plots
    lon = catalog[0].origins[-1].longitude
    lat = catalog[0].origins[-1].latitude
    depth = catalog[0].origins[-1].depth / 1000 * -1  # km
    lat_uncertainty = catalog[0].origins[-1].latitude_errors.uncertainty
    lon_uncertainty = catalog[0].origins[-1].longitude_errors.uncertainty
    z_uncertainty = catalog[0].origins[
                        -1].depth_errors.uncertainty / 1000  # km (does not need to be negative bc abosolute value)
    laterrory = [lat - lat_uncertainty / 110, lat + lat_uncertainty / 110]
    laterrorx = [lon, lon]
    lonerrorx = [lon - lon_uncertainty / 110, lon + lon_uncertainty / 100]
    lonerrory = [lat, lat]
    zerror = [depth - z_uncertainty, depth + z_uncertainty]

    # Plot to horizontal xsection
    if plot_errors:
        fig.axes[1].plot(lonerrorx, [depth, depth], color=color, linewidth=1)
        fig.axes[1].plot([lon, lon], zerror, color=color, linewidth=1)
    fig.axes[1].plot(lon, depth, color=color)

    # Plot to vertical xsection
    if plot_errors:
        fig.axes[2].plot([depth, depth], laterrory, color=color, linewidth=1)
        fig.axes[2].plot(zerror, [lat, lat], color=color, linewidth=1)
    fig.axes[2].plot(depth, lat, color=color)

    return fig


def plot_catalog2xs(fig, catalog, marker='o', color='black', markersize=8, alpha=0.95):

    for eqevent in catalog:
        # Get origin info for xsection plots
        lon = eqevent.origins[-1].longitude
        lat = eqevent.origins[-1].latitude
        depth = eqevent.origins[-1].depth / 1000 * -1  # km
        contains_errors = eqevent.origins[-1].latitude_errors.uncertainty
        if contains_errors:
            lat_uncertainty = eqevent.origins[-1].latitude_errors.uncertainty
            lon_uncertainty = eqevent.origins[-1].longitude_errors.uncertainty
            z_uncertainty = eqevent.origins[
                                -1].depth_errors.uncertainty / 1000  # km (does not need to be negative bc abosolute value)
            laterrory = [lat - lat_uncertainty / 110, lat + lat_uncertainty / 110]
            laterrorx = [lon, lon]
            lonerrorx = [lon - lon_uncertainty / 110, lon + lon_uncertainty / 100]
            lonerrory = [lat, lat]
            zerror = [depth - z_uncertainty, depth + z_uncertainty]

        # Plot to horizontal xsection
        if contains_errors:
            fig.axes[1].plot(lonerrorx, [depth, depth], color=color, linewidth=1)
            fig.axes[1].plot([lon, lon], zerror, color=color, linewidth=1)
        fig.axes[1].plot(lon, depth, color=color)

        # Plot to vertical xsection
        if contains_errors:
            fig.axes[2].plot([depth, depth], laterrory, color=color, linewidth=1)
            fig.axes[2].plot(zerror, [lat, lat], color=color, linewidth=1)
        fig.axes[2].plot(depth, lat, color=color)

    return fig


def plot_hypo2xs(ax, lat=None, lon=None, depth=None, orientation='h', marker='o', color='black', markersize=8, alpha=0.95):

    # Plot to horizontal xsection
    if plot_errors:
        fig.axes[1].plot(lonerrorx, [depth, depth], color=color, linewidth=1)
        fig.axes[1].plot([lon, lon], zerror, color=color, linewidth=1)
    fig.axes[1].plot(lon, depth, color=color)


def plot_eventerror2map(ax, event, color='black', linewidth=1, alpha=0.95,
                        transform=ccrs.Geodetic()) -> object:
    lat = event.origins[-1].latitude
    lon = event.origins[-1].longitude

    # Plot hypocenter uncertainty to map axes as bars (underneath hypocenter)
    lat_uncertainty = event.origins[-1].latitude_errors.uncertainty
    lon_uncertainty = event.origins[-1].longitude_errors.uncertainty
    if lat_uncertainty:
        laty = [lat - lat_uncertainty / 110, lat + lat_uncertainty / 110]
        latx = [lon, lon]
        ax.plot(latx, laty, color=color, linewidth=1)  # plot lat error
    if lon_uncertainty:
        lonx = [lon - lon_uncertainty / 110, lon + lon_uncertainty / 100]
        lony = [lat, lat]
        ax.plot(lonx, lony, color=color, linewidth=1)  # plot lon error

    return ax


def plot_eventerror2xs(ax, event, color='black', linewidth=1, alpha=0.95,
                       transform=ccrs.Geodetic()) -> object:
    pass


########################################################################################################################
### VOLCANO
########################################################################################################################

def plot_volcano(ax, lat, lon, marker='^', color='red', edgecolor='black', markersize=12, alpha=0.95,
                 transform=ccrs.Geodetic(), **kwargs):
    ax.plot(lon, lat, color=edgecolor, linewidth=0, **kwargs)
    return ax


########################################################################################################################
### STATIONS
########################################################################################################################


def plot_stations(ax, lat, lon, marker='v', color='white', edgecolor='black', markersize=10, alpha=0.95,
                  transform=ccrs.Geodetic()):
    ax.plot(lon, lat, linewidth=0)
    return ax


def plot_station_inventory(ax, inventory, marker='v', color='white', edgecolor='black', markersize=6, alpha=0.95,
                           transform=ccrs.Geodetic()):
    import matplotlib.pyplot as plt

    # lat/lon coordinates
    lats = []
    lons = []
    depths = []
    colors = []
    labels = []
    label = 'stub label'

    for net in inventory:
        for sta in net:
            if sta.latitude is None or sta.longitude is None:
                msg = ("Station '%s' does not have latitude/longitude "
                       "information and will not be plotted." % label)
                warnings.warn(msg)
                continue
            label_ = "   " + ".".join((net.code, sta.code))
            color_ = color
            lats.append(sta.latitude)
            lons.append(sta.longitude)
            labels.append(label_)
            colors.append(color_)

    if not label:
        labels = None

    ax.plot(lons, lats, linewidth=0)

    plt.draw()

    return ax


########################################################################################################################
### BACKGROUND MAP
########################################################################################################################

def stamen_background(ax, style='terrain-background', zoom=12):
    # Create a Stamen terrain background instance.
    #    tiler = cimgt.Stamen(style)
    tiler = cimgt.Stamen(style, desired_tile_form='RGBA')
    #    tiler = cimgt.Stamen(style, desired_tile_form='L')
    ax.add_image(tiler, zoom)
    return ax


# TODO change sight_point_pyproj to take km, not m
