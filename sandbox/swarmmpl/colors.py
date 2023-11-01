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
plasma_u = LinearSegmentedColormap.from_list('Upper Half', cm.plasma(np.linspace(-1, 1.2, 256)))
inferno_u = LinearSegmentedColormap.from_list('Upper Half', cm.inferno(np.linspace(-1, 1.2, 256)))
viridis_u = LinearSegmentedColormap.from_list('Upper Half', cm.viridis(np.linspace(-1, 1.2, 256)))