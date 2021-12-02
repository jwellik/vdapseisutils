import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import argparse
import os

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import numpy as np
import datetime

from tables import *

import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.transforms import offset_copy

import vdapseisutils.maputils.utils

plt.rcParams['svg.fonttype'] = 'none'

import time
import shutil
import glob
import urllib

from obspy import UTCDateTime
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel


from bokeh.plotting import figure, output_file, save, gridplot, show
from bokeh.models import HoverTool, ColumnDataSource, OpenURL, TapTool, Range1d, Div, LinearAxis, Span
from bokeh.models import Arrow, VeeHead, ColorBar, LogColorMapper, LogTicker, Span
from bokeh.layouts import column
from bokeh.io import export_svgs

import data

try:
    import urllib2
except:
    pass

import data.grlplotprops as grl# kernel has to be restarted everytime this file is changed


def main(lat, lon, radial_extent_km=50,
         map_type='terrain-background', map_color=True, zoom=9,
         figsize=[10,10]):

    plt.style.use('./utils/eqmap.mplstyle')

    if map_color==True:
        tiles = cimgt.Stamen(map_type)
    else:
        tiles = cimgt.Stamen(map_type, desired_tile_form="L")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=tiles.crs)

    extent = vdapseisutils.maputils.utils.radial_map_extent(lat, lon, radial_extent_km)
    ax.set_extent(extent)

    if map_color==True:
        ax.add_image(tiles, zoom)
    else:
        ax.add_image(tiles, zoom, cmap='Greys_r')

    #ax.tissot(rad_km=20, lons=lon, lats=lat, alpha=0.5, facecolor='none', edgecolor='black', n_samples=180)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=1, color='gray', alpha=0.5)
    # gl = ax.gridlines(crs=tiles.crs, draw_labels=True,
    #                   linewidth=2, color='gray', alpha=0.5)
    gl.xlabels_top = True
    gl.xlabels_bottom = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    plt.draw()

    return ax


if __name__ == '__main__':
    main(data.volcs['Agung']['lat'], data.volcs['Agung']['lon'], map_color=True, zoom=10)

    import vdapseisutils.maputils as maputils

    axm1 = main(data.volcs['Augustine']['lat'], data.volcs['Augustine']['lon'], radial_extent_km=20, zoom=12, map_color=False)
    axm1 = maputils.utils.plot_volcano(axm1, data.volcs['Augustine']['lat'], data.volcs['Augustine']['lon'], transform=ccrs.Geodetic())
    #axm1 = utils.plot_station(data.volcs['Augustine']['stations']['lat'], data.volcs['Augustine']['stations']['lon'])

    main(data.volcs['Erebus']['lat'], data.volcs['Erebus']['lon'], radial_extent_km=300, map_type='watercolor', zoom=8, map_color=True)
    plt.show()