# Developing code to create the colobar for N or Time
# This is a good resource:
# https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import dates as mdates
from obspy import UTCDateTime


def main():
    print('::: COLORBAR :::')

    ####################################################################################################################
    # ALL THIS STUFF WOULD BE IN WINGPLOT
    figsize = (8, 8)  # figsize does affect how the magnitude scale box will look; combine w scatter_scale

    # definitions for the axes (% of figure size)
    bottom, left = 0.10, 0.10
    top, right = 0.1, 0.1
    mwidth, mheight, xsheight = 0.55, 0.55, 0.2  # This is modified
    cbar_height = 0.02  # <-- THIS IS NEW!!!!!!!
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

    """
    NOTE:
    
    When using scatter(), both 'cmap' and 'norm' need to be specified.
    So define those first and then pass each to the colorbar function.

    """


    cbar_type = 'continuous_t'

    if cbar_type == 'continuous':
        cmap = mpl.cm.viridis_r
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cbar_ax, orientation='horizontal', label='N')
    if cbar_type == 'continuous_t':
        cmap = mpl.cm.viridis_r
        norm = mpl.colors.Normalize(vmin=UTCDateTime('1988/03/17').matplotlib_date,
                                    vmax=UTCDateTime('2022/01/05').matplotlib_date)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax, orientation='horizontal', label='Time')
        loc = mdates.AutoDateLocator()  # from matplotlib import dates as mdates
        cb.ax.xaxis.set_major_locator(loc)
        cb.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    elif cbar_type == 'discrete':
        cmap = mpl.cm.magma_r
        bounds = [0, 5, 10, 15, 20, 25, 30]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cbar_ax, orientation='horizontal', spacing='proportional',  # This call to spacing does nothing
                     label="N Earthquakes")
    elif cbar_type == 'discrete2':
        cmap = mpl.cm.magma_r
        bounds = [0, 5, 10, 20, 30, 35]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cbar_ax, orientation='horizontal',
                     spacing='proportional',
                     label="N Earthquakes")
    elif cbar_type == 'magnitude':
        cmap = mpl.cm.magma_r
        bounds = [-1, 0, 1, 2, 3, 4, 5, 6]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cbar_ax, orientation='horizontal',
                     label="Magnitude")
    elif cbar_type == 'listed':
        cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
                .with_extremes(over='0.25', under='0.75'))  # Listed colormap can use with_extremes()

        bounds = [1, 2, 4, 7, 8]  # len(bounds) must be 1 more than len(color_list)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=cbar_ax,
            boundaries=[0] + bounds + [13],  # Adding values for extensions.
            extend='both',
            ticks=bounds,
            spacing='proportional',
            orientation='horizontal',
            label='Discrete intervals, some other units',
        )


    plt.show()


if __name__ == '__main__':
    main()
