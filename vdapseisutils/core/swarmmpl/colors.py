import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


# Colors used by Swarm (https://volcanoes.usgs.gov/software/swarm/download.shtml)
swarm_colors_hex = [
    "#0000ff",
    "#0000cd",
    "#00009b",
    "#000069",
]
swarm_colors_rgba = [
    (0, 0, 255, 255),
    (0, 0, 205, 255),
    (0, 0, 155, 255),
    (0, 0, 105, 255),
]
greyscale_hex = [
    # "#D0D3D4", #"#D7DBDD",
    # "#B3B6B7", #"#E5E7E9",
    # "#979A9A", #"#F2F3F4",
    # "#7B7D7D", #"#F8F9F9",
    "#757575",
    "#616161",
    "#424242",
    "#212121",
]
earthworm_colors_hex = ('#B2000F', '#004C12', '#847200', '#0E01FF')  # Colors used by Earthworm helicorder

# Default colormaps used for spectrograms
# Only use upper half (got the idea from Aaron Wech) of perceptually uniform sequential colormaps
# https://matplotlib.org/stable/gallery/color/colormap_reference.html
#
#    Aaron Wech's colormap
#	colors=cm.jet(np.linspace(-1,1.2,256))
#	color_map = LinearSegmentedColormap.from_list('Upper Half', colors)
#

plasma_u = LinearSegmentedColormap.from_list('plasma_u', cm.plasma(np.linspace(-1, 1.2, 256)))
inferno_u = LinearSegmentedColormap.from_list('inferno_u', cm.inferno(np.linspace(-1, 0.88, 256)))
viridis_u = LinearSegmentedColormap.from_list('viridis_u', cm.viridis(np.linspace(-1, 1.2, 256)))


def plot_colormaps(title="VDAPSEISUTILS COLORMAPS"):
    """PLOT_COLORMAPS Plots custom colormaps alongside Matplotlib colormaps

    Stolen from here: https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Default behavior: Plot pre-defined list of Matplotlib and VDAP colormaps as well as user-defined colormaps

    cmap_list : Optional list of user-defined colormaps
    exclude_vdap_colors : If set to True, this function will only plot the user-defined colormaps
    """

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    vdap_cmap_list = ["jet",
                      "plasma", plasma_u,
                      "inferno", inferno_u,
                      "inferno_r",
                      "viridis", viridis_u,
                      "magma",
                      "cividis"]

    cmap_list = vdap_cmap_list

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(title, fontsize=14)

    for ax, c in zip(axs, cmap_list):

        if isinstance(c, str):
            # If cmap is a string, it is a matplotlib native colormap
            cmap = mpl.colormaps[c]
            name = cmap.name + " (mpl)"  # append "(mpl)" to the name
        else:
            # otherwise, it is a custom defined colormap
            cmap = c
            name = cmap.name

        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    plt.show()
