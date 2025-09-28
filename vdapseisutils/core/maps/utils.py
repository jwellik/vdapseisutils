from __future__ import (absolute_import, division, print_function)

import pandas as pd
import numpy as np

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

import six

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
    for event in catalog:
        lat = event.origins[-1].latitude
        lon = event.origins[-1].longitude
        depth = event.origins[-1].depth / 1000  # convert to km

        # Plot errors
        if plot_errors:
            fig = plot_eventerror2xs(fig, event, color=color, alpha=alpha)

        # Plot to vertical xsection
        fig.axes[2].plot(lat, depth, marker=marker, color=color, markersize=markersize, alpha=alpha)

        # Plot to horizontal xsection
        fig.axes[1].plot(lon, depth, marker=marker, color=color, markersize=markersize, alpha=alpha)

    return fig


def plot_catalog2xs(fig, catalog, marker='o', color='black', markersize=8, alpha=0.95):
    # Get origin info for xsection plots
    for event in catalog:
        lat = event.origins[-1].latitude
        lon = event.origins[-1].longitude
        depth = event.origins[-1].depth / 1000  # convert to km

        # Plot to vertical xsection
        fig.axes[2].plot(lat, depth, marker=marker, color=color, markersize=markersize, alpha=alpha)

        # Plot to horizontal xsection
        fig.axes[1].plot(lon, depth, marker=marker, color=color, markersize=markersize, alpha=alpha)

    return fig


def plot_hypo2xs(ax, lat=None, lon=None, depth=None, orientation='h', marker='o', color='black', markersize=8, alpha=0.95):
    # Plot to horizontal cross-section
    ax.plot(lon, depth, marker=marker, color=color, markersize=markersize, alpha=alpha)
    return ax


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
    import warnings

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