# Developing code to create the magnitude scale in the bottom right of a wingplot
# This link was helpful (size of scatter markers):
# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size

import numpy as np
import matplotlib.pyplot as plt


def main():
    print('::: MAGSCALE :::')

    ####################################################################################################################
    # ALL THIS STUFF WOULD BE IN WINGPLOT
    figsize = (8, 8)  # figsize does affect how the magnitude scale box will look; combine w scatter_scale

    # definitions for the axes (% of figure size)
    bottom, left = 0.05, 0.10
    top, right = 0.1, 0.1
    mwidth, mheight, xsheight = 0.65, 0.65, 0.2
    spacing = 0.005

    map_pos = [left, bottom + xsheight + spacing, mwidth, mheight]
    hxs_pos = [left, bottom, mwidth, xsheight]
    vxs_pos = [left + mwidth + spacing, bottom + xsheight + spacing, xsheight, mheight]
    mag_scale_pos = [left + mwidth + spacing, bottom, xsheight, xsheight]  # <-- THIS IS NEW!!!!!!!!

    # start with a square Figure
    fig = plt.figure(figsize=figsize)
    axm = fig.add_axes(map_pos, title='Test')
    axh = fig.add_axes(hxs_pos)
    axv = fig.add_axes(vxs_pos)
    mag_ax = fig.add_axes(mag_scale_pos)  # <-- THIS IS NEW!!!!!!!!
    ####################################################################################################################



    # STUB - CREATE RANDOM X,Y POINTS for the markers on the map
    map_x = np.random.rand(5)
    map_y = np.random.rand(5)
    map_mag = np.float64(np.random.randint(-1, 6, size=5)) - 0.25
    print('Magnitude: {}'.format(map_mag))


    # Adjust magnitudes from input to work for scatter plotting
    # magnitude scale offset to avoid negative numbers (size can't be negative)
    mso = 0 if min(map_mag) >= 0 else np.floor(np.min(map_mag))*-1
    print("mso: {}".format(mso))

    map_mag += mso  # Magnitude number for for map scatter plot
    scale_mag = np.array([1, 2, 3, 4, 5]) + mso  # Array of magnitudes presented in scale axis
    # scale_mag = np.arange(np.floor(np.min(map_mag)), np.ceil(np.max(map_mag)))
    print('Magnitude_: {}'.format(map_mag))
    print("Scale_Mag_: {}".format(scale_mag))

    scale_type = 'exponential'

    ## size
    # #  -Exponential
    if scale_type == 'exponential':
        scatter_scale = 3  # converts magnitude to scatter plot size (try many numbers across orders of magnitude)
        map_s = scatter_scale ** map_mag  # markersize**2 for the map
        s = scatter_scale ** scale_mag  # markersize**2 for scale box; alternatively, ms=scatter_scale (and use ms in the scatter() function)
    #  -Linear
    elif scale_type == 'linear':
        scatter_scale = 5  # converts magnitude to scatter plot size (try many numbers across orders of magnitude)
        map_s = scatter_scale*map_mag**2  # markersize**2 for the map
        s = scatter_scale*scale_mag**2  # markersize**2 for scale box; alternatively, ms=scatter_scale (and use ms in the scatter() function)
    print("markersize (ms): {}".format(scatter_scale))
    print('size (s): {}'.format(s))

    # MAGNITUDE SCALE
    mag_scale_xpos = np.array([0] * len(scale_mag))  # xpos of mag scale circles is 0, make the array
    if scale_type == 'exponential':
        mag_scale_ypos = scale_mag ** 2  # largest on top, smallest on bottom, 1 order of magnitude apart looks nice
        ylim = (0, (scale_mag[-1] + 2)**2)
    elif scale_type == 'linear':
        mag_scale_ypos = scale_mag * 10
        ylim = (scale_mag[1]*10-15, scale_mag[-1]*10+15)

    # Plot to map and scale box
    axm.scatter(map_x, map_y, s=map_s, color='none', edgecolor='k')  # Plot scatter markers to map [STUB]
    mag_ax.scatter(mag_scale_xpos, y=mag_scale_ypos, s=s, color='none',
                   edgecolor='k')  # Plot scatter makers to scale axis


    print('Mag scale ylim: {}'.format(ylim))
    mag_ax.set_ylim(ylim[0], ylim[1])  # Works best with exponential version
    mag_ax.set_xlim(-0.03, 0.05)  # arbitrarily determined
    mag_ax.set_xticks([])  # remove xticks
    mag_ax.set_yticks(mag_scale_ypos)  # set yticks at height for each circle
    mag_ax.set_yticklabels(['M{}'.format(m-mso) for m in scale_mag])  # give them a label in the format M3, for example
    mag_ax.yaxis.tick_right()  # put yticklabels on the right
    mag_ax.tick_params(axis="y", direction="in", pad=-30, right=False)  # put labels on inside and remove ticks
    [mag_ax.spines[pos].set_visible(False) for pos in ["top", "bottom", "left", "right"]]  # remove axis frames
    mag_ax.patch.set_alpha(0.0)  # set axis background to transparent

    axm.set_ylim(-2, 2)  # stub setting for stub map
    axm.set_xlim(-2, 2)  # stub setting for stub map

    plt.show()


if __name__ == '__main__':
    main()
