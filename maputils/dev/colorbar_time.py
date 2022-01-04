# Developing code to create the magnitude scale in the bottom right of a wingplot
# This link was helpful (size of scatter markers):
# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from obspy import UTCDateTime


def main():
    print('::: COLORBAR (TIME) :::')

    ####################################################################################################################
    # ALL THIS STUFF WOULD BE IN WINGPLOT
    figsize = (6, 6)  # figsize does affect how the magnitude scale box will look; combine w scatter_scale

    # definitions for the axes (% of figure size)
    bottom, left = 0.10, 0.10
    top, right = 0.1, 0.1
    mwidth, mheight, xsheight = 0.55, 0.55, 0.2
    cbar_height = 0.02
    spacing = 0.005

    map_pos = [left, bottom + cbar_height + xsheight + spacing, mwidth, mheight]
    hxs_pos = [left, bottom + cbar_height, mwidth, xsheight]
    vxs_pos = [left + mwidth + spacing, bottom + cbar_height + xsheight + spacing, xsheight, mheight]
    mag_scale_pos = [left + mwidth + spacing, bottom + cbar_height, xsheight, xsheight]  # <-- THIS IS NEW!!!!!!!!
    cbar_pos = [left, 0.08, mwidth + spacing + xsheight, cbar_height]  # <-- THIS IS NEW!!!!!!!!

    # start with a square Figure
    fig = plt.figure(figsize=figsize)
    axm = fig.add_axes(map_pos, title='Test')
    axh = fig.add_axes(hxs_pos)
    axv = fig.add_axes(vxs_pos)
    mag_ax = fig.add_axes(mag_scale_pos)  # <-- THIS IS NEW!!!!!!!!
    cbar_ax = fig.add_axes(cbar_pos)  # <-- THIS IS NEW!!!!!!!!

    ####################################################################################################################


    cmap = mpl.cm.magma_r
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cbar_ax, orientation='horizontal', label='N Earthquakes')

    plt.show()

if __name__ == '__main__':
    main()
