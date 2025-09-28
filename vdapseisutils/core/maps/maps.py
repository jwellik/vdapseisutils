"""
Python scripts for swarmmpl earthquake catalogs at volcanoes.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 26

COMPLETED: Fixed FutureWarning by updating pyproj usage in geodesic_point_buffer functions
TODO LatLonTicks https://cartopy.readthedocs.io/v0.25.0.post2/gallery/gridlines_and_labels/tick_labels.html
TODO Add location map

"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from obspy.imaging.util import _set_xaxis_obspy_dates

from vdapseisutils.utils.geoutils.geoutils import backazimuth
from vdapseisutils.utils.timeutils import convert_timeformat
from vdapseisutils.core.maps import elev_profile
from vdapseisutils.utils.geoutils import sight_point_pyproj, radial_extent2map_extent, project2line

from vdapseisutils.style import load_custom_rc

load_custom_rc("swarmmplrc")

# Plotting styles and formatters for maps and cross-sections
plt.rcParams['svg.fonttype'] = 'none'

# Global font settings - only set base font size, let classes handle specific styling
plt.rcParams['font.size'] = 8  # Base font size (small for crisp appearance)

# Heatmap defaults - change these values to update all plot_heatmap methods
HEATMAP_DEFAULTS = {
    'cmap': 'plasma',
    'alpha': 0.7,
    'grid_size': 0.01,  # Default to 0.05 degrees (≈5km) for VolcanoFigure
    'vmin': None,
    'vmax': None
}

# Comprehensive styling defaults - change these values to update all classes at once
TICK_DEFAULTS = {
    # Tick and ticklabel styling
    'labelcolor': 'grey',
    'labelsize': 'small',  # 0.8x base = 6.4
    'tick_color': 'grey',
    'tick_size': 3,  # tick length
    'tick_width': 1.5,  # tick line width
    'tick_direction': 'out',
    'tick_pad': 2,  # padding between ticks and labels
    
    # Axis label styling
    'axes_labelcolor': 'grey',
    'axes_labelsize': 'medium',  # 1.0x base = 8
    'axes_titlesize': 'large',  # 1.2x base = 9.6
    
    # Legend styling
    'legend_fontsize': 'medium',  # 1.0x base = 8
}

# General axes customization defaults - applies to all axes
AXES_DEFAULTS = {
    # Spine styling
    'spine_linewidth': 1.5,  # thickness of all spines
    'spine_color': 'black',  # color of spines
}

# Cross-section specific defaults
CROSSSECTION_DEFAULTS = {
    # Profile line styling (for cross-sections)
    'profile_linewidth': 1.5,  # linewidth for topographic profiles
    'profile_color': 'k',  # color for topographic profiles
    
    # Text styling for labels and annotations
    'text_stroke_linewidth': 2,  # linewidth for text stroke effects
    'text_stroke_color': 'white',  # color for text stroke effects
    
    # Label positioning
    'ylabel_rotation': 270,  # rotation for y-axis labels
    'ylabel_pad': 15,  # padding for y-axis labels
}

# Grid line styling defaults for maps
GRID_DEFAULTS = {
    'linewidth': 0,
    'color': 'gray',
    'alpha': 0.5,
    'xlines': True,
    'ylabel_style': {'color': 'grey', 'rotation': 90, 'size': 'small'},  # 90 = vertical
    'xlabel_style': {'color': 'grey', 'size': 'small'}
}

# Keep legacy variables for backward compatibility (now using centralized config)
titlefontsize = t1fs = TICK_DEFAULTS['axes_titlesize']
subtitlefontsize = t2fs = TICK_DEFAULTS['axes_labelsize']
axlabelfontsize = axlf = TICK_DEFAULTS['axes_labelsize']
annotationfontsize = afs = plt.rcParams['font.size']
axlabelcolor = axlc = TICK_DEFAULTS['axes_labelcolor']

cmap = "viridis_r"
norm = None

hood = {
    'name': "Hood",
    'synonyms': "Wy'east",
    'lat': 45.374,
    'lon': -121.695,
    'elev': 3426,
}

agung = {
    'name': "Agung",
    'synonyms': "Agung",
    'lat': -8.343,
    'lon': 115.508,
    'elev': 2997,
}

default_volcano = agung


########################################################################################################################
# Development

########################################################################################################################
# TERRAIN UTILS

# deprecated - this won't be used anymore
class ShadedReliefESRI(cimgt.GoogleTiles):
    # shaded relief - produces terrain tile w pink hue and blue water
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url


def add_hillshade_pygmt(ax, extent=[-180, 180, -90, 90],
                        data_source="igpp",
                        resolution="auto",
                        topo=True, bath=False,
                        radiance=[315, 45],
                        vertical_exag=1.5,
                        cmap="Greys_r", alpha=0.8,
                        blend_mode="overlay",
                        elevation_weight=0.3,
                        hillshade_weight=0.7,
                        normalize_elevation=True,
                        cache_data=True,
                        **kwargs,
                        ):
    """
    Add enhanced hillshade and elevation data to a Cartopy GeoAxes using PyGMT.

    See PyGMT's load_earth_relief() for more details:
    - https://www.pygmt.org/latest/_modules/pygmt/datasets/earth_relief.html#load_earth_relief

    This function creates a more visually appealing terrain visualization by combining
    hillshade and elevation data with customizable blending options.

    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The target axes to add the hillshade to
    extent : list
        Map extent [minlon, maxlon, minlat, maxlat]
    data_source : str
        PyGMT data source ("igpp", "srtm", "gebco", etc.)
    resolution : str
        Data resolution ("auto", "01d", "30s", "15s", etc.). If "auto", resolution
        is automatically selected based on map extent.
    topo : bool
        Include topographic (land) data
    bath : bool
        Include bathymetric (ocean) data
    radiance : list
        Hillshade lighting parameters [azimuth, elevation]
    vertical_exag : float
        Vertical exaggeration factor
    cmap : str
        Colormap for elevation data (default: "Greys_r" for black/white)
    alpha : float
        Overall transparency
    blend_mode : str
        Blending mode: "overlay", "multiply", "hillshade_only", "elevation_only"
    elevation_weight : float
        Weight for elevation data in blend (0-1)
    hillshade_weight : float
        Weight for hillshade data in blend (0-1)
    normalize_elevation : bool
        Normalize elevation data for better contrast
    cache_data : bool
        Cache downloaded data for reuse
    transform_data : str
        Data transformation method. Options:
        - "auto": Apply automatic rotation and flip (default)
        - "none": No transformation
        - "rotate_only": Only rotate 90° clockwise
        - "flip_only": Only flip left-to-right
        - "custom": Apply both rotation and flip

    Returns:
    --------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The modified axes
    """
    import numpy as np
    import pygmt
    import matplotlib.colors as mcolors
    from pathlib import Path
    import hashlib

    # Automatic resolution selection based on map extent
    def auto_resolution(extent):
        """
        Automatically select resolution based on map extent.

        Returns appropriate resolution string based on the size of the map area.
        """
        minlon, maxlon, minlat, maxlat = extent
        width_deg = abs(maxlon - minlon)
        height_deg = abs(maxlat - minlat)
        max_dimension = max(width_deg, height_deg)

        # Resolution selection based on map size
        if max_dimension > 180:  # Global or very large regions
            return "01d"
        elif max_dimension > 90:  # Continental scale
            return "30m"
        elif max_dimension > 45:  # Regional scale
            return "15m"
        elif max_dimension > 20:  # Large local areas
            return "03s"
        elif max_dimension > 10:  # Medium local areas
            return "01s"
        elif max_dimension > 5:  # Small local areas
            return "30s"
        elif max_dimension > 1:  # Very small local areas
            return "15s"
        else:  # Tiny areas
            return "03s"

    # Set resolution if auto-selection is requested
    if resolution == "auto":
        resolution = auto_resolution(extent)
        print(f"Auto-selected resolution: {resolution}")

    # Create cache directory
    if cache_data:
        cache_dir = Path.home() / ".vdapseisutils" / "hillshade_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache key from parameters
        cache_params = f"{extent}_{data_source}_{resolution}_{vertical_exag}_{radiance}"
        cache_key = hashlib.md5(cache_params.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.npz"

        # Try to load from cache
        if cache_file.exists():
            try:
                cached_data = np.load(cache_file)
                elevation_data = cached_data['elevation']
                hillshade_data = cached_data['hillshade']
                lats = cached_data['lats']
                lons = cached_data['lons']
                print(f"Loaded hillshade data from cache: {cache_file}")
            except:
                cache_file.unlink(missing_ok=True)
                cached_data = None
        else:
            cached_data = None
    else:
        cached_data = None

    # Download and process data if not cached
    if cached_data is None:
        try:
            # Download elevation data
            srtm = pygmt.datasets.load_earth_relief(
                region=extent,
                data_source=data_source,
                resolution=resolution,
                **kwargs,
            )

            # Apply vertical exaggeration
            elevation_data = srtm.data * vertical_exag

            # Create hillshade
            srtm_hs = pygmt.grdgradient(
                grid=srtm,
                radiance=radiance,
                normalize="t1"  # Normalize to [-1, 1]
            )
            hillshade_data = srtm_hs.data

            # Get coordinate arrays
            lats = srtm.lat.data
            lons = srtm.lon.data

            # Check if latitudes are in descending order (north to south) and flip data if needed
            # PyGMT sometimes returns data with latitudes in descending order, but matplotlib expects ascending
            if len(lats) > 1 and lats[0] > lats[-1]:
                print("Flipping data vertically to match matplotlib's expected coordinate system")
                elevation_data = np.flipud(elevation_data)
                hillshade_data = np.flipud(hillshade_data)
                lats = np.flipud(lats)

            # Check if longitudes need wrapping for regions crossing the 180/-180 meridian
            if len(lons) > 1 and (lons[0] < -170 and lons[-1] > 170):
                print("Detected region crossing 180/-180 meridian, adjusting longitude ordering")
                # For regions crossing the meridian, we might need to roll the data
                # This is a complex case that might need custom handling

            # # Debug: Print coordinate ranges to help diagnose orientation issues
            # print(f"Data shape: {elevation_data.shape}")
            # print(f"Latitude range: {lats.min():.3f} to {lats.max():.3f}")
            # print(f"Longitude range: {lons.min():.3f} to {lons.max():.3f}")
            # print(f"Latitude ordering: {'descending' if lats[0] > lats[-1] else 'ascending'}")
            # print(f"Longitude ordering: {'descending' if lons[0] > lons[-1] else 'ascending'}")

            # Cache the data
            if cache_data:
                np.savez(cache_file,
                         elevation=elevation_data,
                         hillshade=hillshade_data,
                         lats=lats,
                         lons=lons)
                print(f"Cached hillshade data: {cache_file}")

        except Exception as e:
            print(f"Failed to load hillshade data: {e}")
            return ax

    # Apply land/ocean masking
    if not bath:
        hillshade_data[elevation_data <= 0] = 1.0  # White for ocean
        elevation_data[elevation_data <= 0] = 0
    if not topo:
        hillshade_data[elevation_data >= 0] = 1.0  # White for land
        elevation_data[elevation_data >= 0] = 0

    # Create custom cream/white colormap for better appearance
    def create_cream_colormap():
        """Create a custom colormap with cream background and black features."""
        colors = ['#FDFBF7', '#F5F1E8', '#E8E0D0', '#D4C8B8', '#C0B0A0',
                  '#A89888', '#908070', '#786858', '#605040', '#483828', '#302010']
        return mcolors.LinearSegmentedColormap.from_list('cream_terrain', colors)

    # Prepare data for plotting
    if blend_mode == "hillshade_only":
        # Pure hillshade - convert to grayscale
        final_data = (hillshade_data + 1) / 2  # Convert from [-1,1] to [0,1]
        plot_cmap = "Greys_r"

    elif blend_mode == "elevation_only":
        # Pure elevation with colormap
        final_data = elevation_data
        if cmap == "Greys_r":
            plot_cmap = create_cream_colormap()
        else:
            plot_cmap = cmap

    elif blend_mode == "multiply":
        # Multiply hillshade with elevation
        hillshade_norm = (hillshade_data + 1) / 2  # Convert to [0,1]
        if normalize_elevation:
            elev_norm = (elevation_data - elevation_data.min()) / (elevation_data.max() - elevation_data.min())
        else:
            elev_norm = elevation_data / elevation_data.max()
        final_data = hillshade_norm * elev_norm
        if cmap == "Greys_r":
            plot_cmap = create_cream_colormap()
        else:
            plot_cmap = cmap

    else:  # "overlay" mode (default)
        # Blend hillshade and elevation
        hillshade_norm = (hillshade_data + 1) / 2  # Convert to [0,1]

        if normalize_elevation:
            elev_norm = (elevation_data - elevation_data.min()) / (elevation_data.max() - elevation_data.min())
        else:
            elev_norm = elevation_data / elevation_data.max()

        # Create blended image
        final_data = hillshade_weight * hillshade_norm + elevation_weight * elev_norm
        if cmap == "Greys_r":
            plot_cmap = create_cream_colormap()
        else:
            plot_cmap = cmap

    plot_data = np.flipud(final_data)

    projection = ccrs.PlateCarree()
    ax.imshow(
        plot_data,
        extent=extent,
        transform=projection,
        cmap=plot_cmap,
        alpha=alpha,
        interpolation='bilinear'
    )

    return ax


########################################################################################################################
# SCALE BAR

# Distance (km) to _____?
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


# Distance (km) to axes size
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


##############################################################################################################s
# Misc Axes

class MagLegend:

    # Scale the magnitudes to marker size and scatter plot size
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    # default rcParams['line.markersize'] = 6
    # default scatter size is rcParams['line.markersize']**2
    # markersize is equivalent to line.markersize
    # size is equivalent to line.markersize**2 (the size of the scatter dot)
    # default values make a M0 roughly the default marker size
    # mrange
    # Out: array([-2, -1, *0*, 1, 2, 3, 4, 5])
    # msrange
    # Out: array([0., 2.85714286, *5.71428571*, 8.57142857, 11.42857143,
    #        14.28571429, 17.14285714, 20.])

    def __init__(self,
                 # mrange=[-2, 5], msrange=[0, 15],  # results in M0 swarmmpl at ~markersize=6, default (see above)
                 mrange=[-2, 2], msrange=[0, 6],
                 # defined this way so M-2 is smallest possible event & M2 is ~markersize=6 (default)
                 disprange=[-1, 5]
                 ):

        self.mrange = np.arange(mrange[0], mrange[1] + 1)  # array of magnitudes for the legend
        self.msrange = np.linspace(start=msrange[0], stop=msrange[1], num=len(self.mrange))  # array of marker sizes
        self.srange = self.msrange ** 2  # range of sizes in points (markersize**2)

        self.legend_mag = np.arange(disprange[0], disprange[1] + 1)  # array of magnitudes to be plotted on legend
        self.legend_s = self.mag2s(self.legend_mag)  # array of circle sizes on legend corresponding to mag

        self.n = len(self.mrange)

    def legend_scale(self, color="k", alpha=1.0):
        fig = plt.figure()
        plt.scatter(self.mrange, self.msrange, s=self.srange, color=color, alpha=alpha)
        plt.show()

    def display(self, ax=None, color="none", edgecolor="k", include_counts=True):

        if ax == None:
            fig, ax = plt.subplots()

        ax.scatter([0] * len(self.legend_mag), y=self.legend_mag, s=self.legend_s,
                   color=color, edgecolor=edgecolor)

        # Change settings on scale box axes
        # ax.set_ylim(self.legend_mag[0]-0.5, self.legend_mag[1]+1.5)  # Just guessing
        # ax.set_xlim(-0.02, 0.02)  # arbitrarily determined
        ax.set_xticks([])  # remove xticks
        ax.set_yticks(self.legend_mag)  # set yticks at height for each circle
        ax.set_yticklabels(['M{}'.format(m) for m in self.legend_mag])  # no counts
        # ax.set_yticklabels(['M{} ({} eqs)'.format(m, n) for m, n in
        #                      zip(self.legend_mag, self.legend_counts([]))])  # give them a label in the format M3, for example
        ax.yaxis.tick_right()  # put yticklabels on the right
        ax.tick_params(axis="y", direction="out", pad=0, right=False)  # put labels on inside and remove ticks
        ax.patch.set_alpha(0.0)  # set axis background to transparent

        # Set thicker spines for visible spines
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(AXES_DEFAULTS['spine_linewidth'])

        ax.spines['top'].set_visible(False)  # make all spines invisible
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if include_counts:
            pass

        return ax

    def mag2s(self, mag):
        """MAG2S Converts magnitude to point size for scatter plot
        Uses ranges set for magnitude and marksersize range
        Point size is markersize**2
        Default is M0 is roughly equal to default markersize of 6 (point size 36)
        """
        mag = np.array(mag)  # ensure mag is an array
        m, b = np.polyfit(self.mrange, self.msrange, 1)
        ms = m * mag + b  # m*mag+b converts to marker size
        ms[ms < 0] = 0  # marker size must be >=0
        s = ms ** 2  # convert to point size (**2)
        return s

    def legend_counts(self, cat):
        """COUNTS Counts the number of EQs at each magnitude within the legend scale"""

        nmags = []
        for mag in self.legend_mag:
            rslt = cat[(cat["mag"] >= mag) & (cat["mag"] < mag + 1)]
            nmags.append(len(rslt))

        return nmags

    def info(self):

        print("::: Magnitude Legend Information :::")
        print("     ms: markersize (default=6)")
        print("     s : point size (markersize**2")
        for M, ms, s in zip(self.mrange, self.msrange, self.srange):
            print("M{:>-4.1f} | ms: {:>4.1f} | s: {:>4.1f}".format(M, ms, s))
        print()


class ColorBar:
    pass


##############################################################################################################s
# Misc utils

def prep_catalog_data_mpl(catalog, s="magnitude", c="time", maglegend=MagLegend(), time_format="matplotlib"):
    """ PREPCATALOG Converts ObsPy Catalog object to DataFrame w fields appropriate for swarmmpl

    TODO Allow for custom MagLegends
    TODO Add color column
    TODO Filter catalog to extents and return nRemoved

    :return:
    """

    ## Get info out of Events object
    from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm
    import pandas as pd
    # returns time(UTCDateTime), lat, lon, depth(km, positive below sea level), mag
    catdata = catalog2txyzm(catalog, time_format=time_format)
    catdata = pd.DataFrame(catdata).sort_values("time")
    catdata["depth"] *= -1  # below sea level values are negative for swarmmpl purposes
    catdata["size"] = MagLegend().mag2s(catdata["mag"])  # converts magnitudes to point size for scatter plot
    return catdata


class Map:
    """
    Map class that creates map axes without inheriting from plt.Figure
    This avoids conflicts when used with SubFigures
    """
    
    name = "map"

    def __init__(self, fig=None, origin=(default_volcano["lat"], default_volcano["lon"]), 
                 radial_extent_km=50.0, map_extent=None, **kwargs):
        
        # Create figure if none provided
        if fig is None:
            # Extract figure-specific kwargs
            fig_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['dpi', 'figsize']}
            fig = plt.figure(**fig_kwargs)
        
        # Store the figure reference
        self.figure = fig
        
        # Remove any matplotlib figure-specific kwargs that would cause issues
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['dpi', 'figsize']}
        
        # Create the principal axes for the map
        self.ax = fig.add_subplot(111, projection=ccrs.Mercator(), **plot_kwargs)
        
        # Set thicker spines
        for spine in self.ax.spines.values():
            spine.set_linewidth(AXES_DEFAULTS['spine_linewidth'])

        # Define Map Properties
        self.properties = dict()
        self.properties["origin"] = origin
        self.properties["radial_extent_km"] = radial_extent_km
        self.properties["map_extent"] = radial_extent2map_extent(origin[0], origin[1], radial_extent_km)
        if map_extent:  # overwrite origin, radial_extent_km if map_extent is set explicitly
            self.properties["origin"] = None
            self.properties["radial_extent_km"] = None
            self.properties["map_extent"] = map_extent

        # Set extent and add labels
        self.ax.set_extent(self.properties["map_extent"])

        # Draw Grid and labels using centralized styling
        glv = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                               linewidth=GRID_DEFAULTS['linewidth'], 
                               color=GRID_DEFAULTS['color'], 
                               alpha=GRID_DEFAULTS['alpha'])
        glv.top_labels = False
        glv.bottom_labels = True
        glv.left_labels = True
        glv.right_labels = False
        glv.xlines = GRID_DEFAULTS['xlines']
        glv.xlabel_style = GRID_DEFAULTS['xlabel_style']
        glv.ylabel_style = GRID_DEFAULTS['ylabel_style']

        # Apply centralized tick styling
        self.ax.tick_params(axis='both', 
                            labelcolor=TICK_DEFAULTS['labelcolor'],
                            labelsize=TICK_DEFAULTS['labelsize'],
                            color=TICK_DEFAULTS['tick_color'],
                            length=TICK_DEFAULTS['tick_size'],
                            width=TICK_DEFAULTS['tick_width'],
                            direction=TICK_DEFAULTS['tick_direction'],
                            pad=TICK_DEFAULTS['tick_pad'])

    # Keep all the other methods from your original Map class
    def info(self):
        print("::: MAP AXES :::")
        print(self.properties)
        print()

    def set_ticks(self, x_spacing=None, y_spacing=None, 
                 show_bottom=True, show_left=True, 
                 show_top=False, show_right=False):
        """
        Customize the gridliner ticks on the map.
        
        Parameters:
        -----------
        x_spacing : float, optional
            Spacing for x-axis (longitude) ticks in degrees
        y_spacing : float, optional  
            Spacing for y-axis (latitude) ticks in degrees
        show_bottom : bool, optional
            Show ticks on bottom axis (default: True)
        show_left : bool, optional
            Show ticks on left axis (default: True)
        show_top : bool, optional
            Show ticks on top axis (default: False)
        show_right : bool, optional
            Show ticks on right axis (default: False)
        """
        # Get the gridliner object
        glv = None
        for child in self.ax.get_children():
            if hasattr(child, 'xlocator'):  # This identifies the gridliner
                glv = child
                break
        
        if glv is not None:
            if x_spacing is not None:
                glv.xlocator = plt.MultipleLocator(x_spacing)
            if y_spacing is not None:
                glv.ylocator = plt.MultipleLocator(y_spacing)
            
            # Set which sides show labels
            glv.xlabels_bottom = show_bottom
            glv.ylabels_left = show_left
            glv.xlabels_top = show_top
            glv.ylabels_right = show_right

    def add_hillshade(self, source="PyGMT", data_source="igpp", resolution="auto", 
                     topo=True, bath=False, radiance=[315, 60], alpha=0.8,
                     blend_mode="overlay", elevation_weight=0.3, hillshade_weight=0.7,
                     cmap="Greys_r", vertical_exag=1.5, normalize_elevation=True,
                     cache_data=True, **kwargs):
        if source.lower() == "pygmt":
            try:
                self.ax = add_hillshade_pygmt(
                    self.ax, 
                    extent=self.properties["map_extent"], 
                    data_source=data_source,
                    resolution=resolution, 
                    topo=topo, 
                    bath=bath,
                    radiance=radiance, 
                    alpha=alpha,
                    blend_mode=blend_mode,
                    elevation_weight=elevation_weight,
                    hillshade_weight=hillshade_weight,
                    cmap=cmap,
                    vertical_exag=vertical_exag,
                    normalize_elevation=normalize_elevation,
                    cache_data=cache_data,
                    **kwargs,
                )
            except Exception as e:
                print(f"Failed to add hillshade via PyGMT: {e}")
                print("Continuing without hillshade...")
        else:
            print(f"Elevation & hillshade source '{source}' not available.")

    def add_scalebar(self, scale_length_km="auto", position='lower right', 
                     color='black', fontsize=10, pad=0.5, frameon=False):
        """
        Add a scale bar to the map.
        
        Creates a scale bar showing the distance scale of the map. The scale bar
        length can be automatically calculated based on the map extent or manually
        specified. The text label appears on top of the scale bar.
        
        Parameters:
        -----------
        scale_length_km : int, float, or str, optional
            Length of the scale bar in kilometers. If "auto" (default), the length
            is automatically calculated based on the map width using 
            choose_scale_bar_length().
        position : str, optional
            Position of the scale bar on the map. Options: 'lower right', 
            'lower left', 'upper right', 'upper left', 'center right', etc.
            (default: 'lower right')
        color : str, optional
            Color of the scale bar and text (default: 'black')
        fontsize : int, optional
            Font size of the scale bar text label (default: 10)
        pad : float, optional
            Padding around the scale bar in axes coordinates (default: 0.5)
        frameon : bool, optional
            Whether to draw a frame around the scale bar (default: False)
            
        Notes:
        ------
        The scale bar length is calculated as a fraction of the map width.
        For automatic calculation, the function uses 25% of the map width as
        a target and selects the largest standard scale bar length that is
        less than or equal to the target from a predefined list of common values.
        
        The scale bar is positioned in axes coordinates, so it will maintain
        its relative position even if the map extent changes.
        """
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm
        from vdapseisutils.utils.geoutils import backazimuth

        # Get scalebar length in axes percentage
        map_lonL, map_lonR, map_midlat = self.properties["map_extent"][0:3]
        az, d = backazimuth((map_midlat, map_lonL), (map_midlat, map_lonR))
        map_width_km = d / 1000
        
        if scale_length_km == "auto":
            scale_length_km = choose_scale_bar_length(map_width_km, 0.25)
        
        scale_bar_length_ax = scale_length_km / map_width_km

        # Determine appropriate unit and label
        if scale_length_km < 1:
            # Convert to meters for small scales
            scale_length_m = int(scale_length_km * 1000)
            scale_label = f"{scale_length_m} m"
        else:
            # Use kilometers for larger scales
            if scale_length_km == int(scale_length_km):
                scale_label = f"{int(scale_length_km)} km"
            else:
                scale_label = f"{scale_length_km} km"

        # Add scale bar with text on top
        scalebar = AnchoredSizeBar(
            self.ax.transAxes,
            scale_bar_length_ax,
            scale_label,
            position,
            pad=pad,
            color=color,
            frameon=frameon,
            size_vertical=0.01,
            fontproperties=fm.FontProperties(size=fontsize),
            label_top=True  # This puts the text on top of the scale bar
        )

        self.ax.add_artist(scalebar)

    def plot(self, lat, lon, *args, transform=ccrs.Geodetic(), **kwargs):
        self.ax.plot(lon, lat, *args, transform=transform, **kwargs)

    def scatter(self, lat, lon, size, color, transform=ccrs.Geodetic(), **kwargs):
        self.ax.scatter(lon, lat, size, color, transform=transform, **kwargs)

    def plot_catalog(self, catalog, s="magnitude", c="time", color=None, cmap="viridis_r", alpha=0.5, transform=ccrs.Geodetic(), **kwargs):
        """
        Plot earthquake catalog on the map.
        
        Creates a scatter plot of earthquake events from an ObsPy Catalog object,
        with customizable size, color, and styling options.
        
        Parameters:
        -----------
        catalog : obspy.core.event.Catalog
            ObsPy Catalog object containing earthquake events
        s : str or array-like, optional
            Size parameter for scatter points. If "magnitude" (default), 
            point sizes are scaled by earthquake magnitude. Otherwise, 
            can be a numeric array or column name from catalog data.
        c : str or array-like, optional
            Color parameter for scatter points. If "time" (default), 
            points are colored by event time. Otherwise, can be a numeric 
            array, column name, or color specification.
        color : str or array-like, optional
            Alternative to 'c' parameter for specifying color. If provided,
            takes precedence over 'c'. Can be a single color name, hex code,
            or array of colors. Compatible with matplotlib's scatter color
            parameter.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for mapping numeric values to colors (default: "viridis_r")
        alpha : float, optional
            Transparency of scatter points, 0 (transparent) to 1 (opaque) 
            (default: 0.5)
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for the data (default: ccrs.Geodetic())
        **kwargs
            Additional keyword arguments passed to matplotlib's scatter function
            
        Returns:
        --------
        matplotlib.collections.PathCollection
            The scatter plot collection
            
        Notes:
        ------
        - Point sizes are automatically scaled based on magnitude using the
          MagLegend class when s="magnitude"
        - Colors are mapped using the specified colormap when c or color 
          contains numeric data
        - Both 'c' and 'color' parameters are supported for compatibility
          with matplotlib's scatter function
        - The method automatically extracts latitude, longitude, magnitude,
          and time data from the ObsPy Catalog object
          
        Examples:
        ---------
        # Basic usage with default magnitude sizing and time coloring
        map_obj.plot_catalog(catalog)
        
        # Custom size and color
        map_obj.plot_catalog(catalog, s="magnitude", c="depth", cmap="plasma")
        
        # Use color parameter instead of c
        map_obj.plot_catalog(catalog, color="red", alpha=0.8)
        
        # Custom size array and color mapping
        map_obj.plot_catalog(catalog, s=[10, 20, 30], c=[1, 2, 3], cmap="cool")
        """
        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        # Handle both 'c' and 'color' parameters like matplotlib's scatter
        if color is not None:
            # 'color' parameter takes precedence over 'c'
            c = color
        elif c == "time":
            c = catdata["time"]
        else:
            c = c

        return self.ax.scatter(catdata["lon"], catdata["lat"], s=s, c=c, cmap=cmap, alpha=alpha, transform=transform, **kwargs)

    def plot_line(self, p1, p2, color="k", linewidth=1,
                  label=None, va='center', ha='center',
                  transform=ccrs.Geodetic(), **kwargs):
        label0 = label if label else ""
        label1 = label+"'" if label else ""
        self.ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=linewidth,
                     transform=transform, **kwargs)
        self.ax.text(p1[1], p1[0], label0, ha=ha, va=va, color=color,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                     transform=transform)
        self.ax.text(p2[1], p2[0], label1, ha=ha, va=va, color=color,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                     transform=transform)

    def plot_inventory(self, inventory, marker_size=8, color='black', alpha=0.8, 
                      transform=ccrs.Geodetic(), **kwargs):
        try:
            station_lats = []
            station_lons = []
            
            for network in inventory:
                for station in network:
                    if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                        station_lats.append(station.latitude)
                        station_lons.append(station.longitude)
            
            if station_lats and station_lons:
                self.ax.scatter(station_lons, station_lats, 
                              s=marker_size, 
                              c=color, 
                              marker='v',
                              alpha=alpha,
                              transform=transform,
                              **kwargs)
            else:
                print("No valid station coordinates found in inventory")
                
        except Exception as e:
            print(f"Error plotting inventory: {e}")
            print("Continuing without inventory plot...")

    def plot_heatmap(self, *args, grid_size=HEATMAP_DEFAULTS['grid_size'], 
                     cmap=HEATMAP_DEFAULTS['cmap'], alpha=HEATMAP_DEFAULTS['alpha'], 
                     vmin=HEATMAP_DEFAULTS['vmin'], vmax=HEATMAP_DEFAULTS['vmax'], **kwargs):
        try:
            # Handle input data based on first argument type
            if len(args) == 1 and hasattr(args[0], 'events'):  # Single Catalog object
                catalog = args[0]
                catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
                lat = catdata["lat"].values
                lon = catdata["lon"].values
                depth = catdata["depth"].values
            elif len(args) >= 2:  # lat, lon, [depth]
                lat = np.asarray(args[0])
                lon = np.asarray(args[1])
                depth = np.asarray(args[2]) if len(args) > 2 else None
            else:
                raise ValueError("Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, [depth], ...)")
            
            if len(lat) == 0 or len(lon) == 0:
                print("Warning: Empty coordinate arrays")
                return None
            
            # Create regular grid
            lon_min, lon_max = np.min(lon), np.max(lon)
            lat_min, lat_max = np.min(lat), np.max(lat)
            
            grid_size_deg = grid_size
            if grid_size_deg < 0.001:
                grid_size_deg = 0.001
            
            data_range_lon = lon_max - lon_min
            data_range_lat = lat_max - lat_min
            min_grid_size = max(0.001, min(data_range_lon, data_range_lat) * 0.1)
            if grid_size_deg < min_grid_size:
                grid_size_deg = min_grid_size
            
            if lon_max <= lon_min or lat_max <= lat_min:
                print("Warning: Invalid coordinate ranges for heatmap")
                return None
            
            lon_pad = (lon_max - lon_min) * 0.1
            lat_pad = (lat_max - lat_min) * 0.1
            
            lon_grid = np.arange(lon_min - lon_pad, lon_max + lon_pad, grid_size_deg)
            lat_grid = np.arange(lat_min - lat_pad, lat_max + lat_pad, grid_size_deg)
            
            if len(lon_grid) < 2 or len(lat_grid) < 2:
                print("Warning: Grid too small for heatmap")
                return None
            
            H, xedges, yedges = np.histogram2d(lon, lat, bins=[lon_grid, lat_grid])
            
            if H.size == 0 or np.all(H == 0):
                print("Warning: No data points in the specified region")
                return None
            
            lon_centers = (xedges[:-1] + xedges[1:]) / 2
            lat_centers = (yedges[:-1] + yedges[1:]) / 2
            lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
            
            im = self.ax.pcolormesh(lon_mesh, lat_mesh, H.T, 
                                   cmap=cmap, alpha=alpha, 
                                   vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(),
                                   **kwargs)
            
            return im
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            print("Continuing without heatmap...")
            return None

    def add_terrain(self, zoom='auto', cache=False):
        """
        Add terrain background tiles from default source to the map.

        Parameters:
        -----------
        zoom : int or str, optional
            Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
        cache : bool, optional
            Whether to cache tiles (default: False)
        """
        self.add_arcgis_terrain(zoom=zoom, cache=cache)

    def add_arcgis_terrain(self, zoom='auto', style='terrain', cache=False):
        """
        Add world terrain background tiles from ArcGIS to the map.

        Parameters:
        -----------
        zoom : int or str, optional
            Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
        style : str, optional
            Style of ArcGIS tiles ('terrain', 'street', 'satellite', default: 'terrain')
        cache : bool, optional
            Whether to cache tiles (default: False)
        """
        from .map_tiles import add_arcgis_terrain
        
        add_arcgis_terrain(
            self.ax, 
            zoom=zoom, 
            style=style, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km")
        )

    def add_google_terrain(self, zoom='auto', cache=False, **kwargs):
        """
        Add Google terrain tiles to the map.
        
        Parameters:
        -----------
        zoom : int or str, optional
            Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
        cache : bool, optional
            Whether to cache tiles (default: False)
        """
        from .map_tiles import add_google_terrain
        
        add_google_terrain(
            self.ax, 
            zoom=zoom, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km"),
            **kwargs
        )

    def add_google_street(self, zoom='auto', cache=False, **kwargs):
        """
        Add Google street tiles to the map.
        
        Parameters:
        -----------
        zoom : int or str, optional
            Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
        cache : bool, optional
            Whether to cache tiles (default: False)
        """
        from .map_tiles import add_google_street
        
        add_google_street(
            self.ax, 
            zoom=zoom, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km"),
            **kwargs
        )

    def add_google_satellite(self, zoom='auto', cache=False, **kwargs):
        """
        Add Google satellite tiles to the map.
        
        Parameters:
        -----------
        zoom : int or str, optional
            Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
        cache : bool, optional
            Whether to cache tiles (default: False)
        """
        from .map_tiles import add_google_satellite
        
        add_google_satellite(
            self.ax, 
            zoom=zoom, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km"),
            **kwargs
        )

    def add_world_location_map(self, size=0.18, position='upper left', **kwargs):
        """
        Add a world location reference map as an inset to the main map.
        
        Creates a small circular world map in the top left corner (or specified position)
        that shows the global context of the main map. The world map is centered on
        the same longitude as the main map center and shows a square marker for the main map center.
        
        Parameters:
        -----------
        size : float, optional
            Size of the inset map as fraction of figure (default: 0.18)
        position : str, optional
            Position of the inset map ('upper left', 'upper right', 'lower left', 'lower right')
            (default: 'upper left')
        **kwargs
            Additional keyword arguments passed to make_world_location_map()
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object for the world location map inset
        """
        # Calculate center coordinates from main map extent
        map_extent = self.properties["map_extent"]
        center_lon = (map_extent[0] + map_extent[1]) / 2  # Average of min and max longitude
        center_lat = (map_extent[2] + map_extent[3]) / 2  # Average of min and max latitude
        
        # Get main map center for marking
        if self.properties["origin"] is not None:
            main_map_center = self.properties["origin"]
        else:
            main_map_center = (center_lat, center_lon)
                        
        # Calculate the position for the inset axes relative to the main axes
        fig = self.ax.figure
        
        # Get the main axes position in figure coordinates
        main_ax_pos = self.ax.get_position()
        main_left = main_ax_pos.x0
        main_bottom = main_ax_pos.y0
        main_width = main_ax_pos.width
        main_height = main_ax_pos.height
        
        # Calculate position relative to main axes
        if position == 'upper left':
            left = main_left - (size/2) * main_width # + 0.02 * main_width
            bottom = main_bottom + main_height - (size/2) * main_height - 0.05 * main_height
        elif position == 'upper right':
            left = main_left + main_width - (size/2) - 0.02 * main_width
            bottom = main_bottom + main_height - (size/2) * main_height - 0.05 * main_height
        elif position == 'lower left':
            left = main_left - (size/2) * main_width # + 0.02 * main_width
            bottom = main_bottom - (size/2) * main_height + 0.02 * main_height
        elif position == 'lower right':
            left = main_left + main_width - (size/2) - 0.02 * main_width
            bottom = main_bottom - (size/2) * main_height + 0.02 * main_height
        else:  # lower left
            left = main_left - (size/2) * main_width + 0.02 * main_width
            bottom = main_bottom + main_height - (size/2) * main_height - 0.05 * main_height
        
        # Create GeoAxes directly with orthographic projection
        world_ax = fig.add_axes([left, bottom, size, size], 
                               projection=ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat))
        
        # Set global extent
        world_ax.set_global()
        
        # Add features with specified styling
        # Grey oceans
        world_ax.add_feature(cfeature.OCEAN, color='lightgrey', alpha=0.8)
        
        # White land
        world_ax.add_feature(cfeature.LAND, color='white', alpha=1.0)
        
        # Country borders
        world_ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Coastlines
        world_ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add grid
        gl = world_ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                               linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlines = True
        gl.ylines = True
        
        # Add square marker for main map center if provided
        if main_map_center is not None:
            main_lat, main_lon = main_map_center
            world_ax.plot(main_lon, main_lat, 's', color='black', markersize=5, 
                         transform=ccrs.Geodetic())
        
        # Remove axis labels and ticks
        world_ax.set_xticks([])
        world_ax.set_yticks([])
        
        # Make the plot circular by setting equal aspect ratio
        world_ax.set_aspect('equal')
        
        return world_ax


class CrossSection:
    """
    CrossSection class that creates cross-section axes without inheriting from plt.Figure
    This avoids conflicts when used with SubFigures
    """

    name = "cross-section"

    def __init__(self, fig=None, points=[(46.198776, -122.261317), (46.197484, -122.122234)],
                 origin=None, radius_km=25.0, azimuth=270, map_extent=None,
                 depth_extent=(-50., 4.), resolution="auto", max_n=100,
                 label="A", width=None, maglegend=MagLegend(), verbose=False, **kwargs):

        # Create figure if none provided
        if fig is None:
            # Extract figure-specific kwargs
            fig_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['dpi', 'figsize']}
            fig = plt.figure(**fig_kwargs)

        # Store the figure reference
        self.figure = fig
        
        # Remove any matplotlib figure-specific kwargs that would cause issues
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['dpi', 'figsize']}
        
        # Create the axes
        self.ax = fig.add_subplot(111, **plot_kwargs)
        
        # Set thicker spines
        for spine in self.ax.spines.values():
            spine.set_linewidth(AXES_DEFAULTS['spine_linewidth'])

        # Better organization
        self.properties = dict()

        # Determine A & A' locations
        if origin:
            self.properties["origin"] = origin
            self.properties["azimuth"] = azimuth
            self.properties["radius"] = radius_km * 1000
            self.properties["points"] = [np.nan, np.nan]
            self.properties["points"][0] = sight_point_pyproj(origin, azimuth, self.properties["radius"])
            self.properties["points"][1] = sight_point_pyproj(origin, np.mod(azimuth + 180, 360), self.properties["radius"])
        else:
            if len(points) != 2:
                raise ValueError("ERROR: Points must be a list of 2 tuples of lat,lon coordinates.")
            self.properties["points"] = points
            self.properties["origin"] = None
            self.properties["azimuth"], self.properties["length"] = backazimuth(points[0], points[1])
            self.properties["radius"] = None

        # Extents by which to filter input data
        self.properties["map_extent"] = map_extent
        self.properties["depth_extent"] = depth_extent
        self.properties["depth_range"] = depth_extent[1] - depth_extent[0]
        self.properties["label"] = label
        self.properties["full_label"] = "{a}-{a}'".format(a=self.properties["label"])
        self.properties["width"] = width
        self.properties["orientation"] = "horizontal"

        self.verbose = verbose

        # Stub (I probably shouldn't redefine these here)
        self.A1 = self.properties["points"][0]
        self.A2 = self.properties["points"][1]

        self.profile = elev_profile.TopographicProfile([self.A1, self.A2], resolution=resolution, max_n=max_n)
        if np.any(self.profile.elevation):
            self.__add_profile()

        self.__setup_axis_formatting()
        self.__add_labels_to_xsection()

    def __add_profile(self):
        hd = self.profile.distance / 1000  # horizontal distance along line (convert meters to km)
        elev = np.array(self.profile.elevation / 1000)  # elevation (convert m to km)
        self.plot(x=hd, z=elev, z_dir="elev", color=CROSSSECTION_DEFAULTS['profile_color'], z_unit="km", linewidth=CROSSSECTION_DEFAULTS['profile_linewidth'])
        self.set_depth_extent()
        self.set_horiz_extent()

        # custom spine bounds for a nice clean look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines["left"].set_bounds((self.properties["depth_extent"][0], elev[0]))
        self.ax.spines["right"].set_bounds(self.properties["depth_extent"][0], elev[-1])

    def __setup_axis_formatting(self):
        # Apply centralized tick styling
        self.ax.tick_params(axis='both', 
                            labelcolor=TICK_DEFAULTS['labelcolor'],
                            labelsize=TICK_DEFAULTS['labelsize'],
                            color=TICK_DEFAULTS['tick_color'],
                            length=TICK_DEFAULTS['tick_size'],
                            width=TICK_DEFAULTS['tick_width'],
                            direction=TICK_DEFAULTS['tick_direction'],
                            pad=TICK_DEFAULTS['tick_pad'],
                            left=False, labelleft=False,
                            bottom=True, labelbottom=True,
                            right=True, labelright=True,
                            top=False, labeltop=False)
        self.ax.yaxis.set_label_position("right")
        self.ax.set_ylabel("Depth (km)", rotation=CROSSSECTION_DEFAULTS['ylabel_rotation'], labelpad=CROSSSECTION_DEFAULTS['ylabel_pad'], 
                           color=TICK_DEFAULTS['axes_labelcolor'], 
                           fontsize=TICK_DEFAULTS['axes_labelsize'])
        
        # Remove xlabel
        self.ax.set_xlabel("")  # Remove xlabel
    
    def _append_km_to_last_xtick_dep001(self):
        """Append ' km' to the last xticklabel while preserving original formatting."""
        # Force matplotlib to draw the plot first to ensure ticks are generated
        self.ax.figure.canvas.draw()
        
        # Get current tick labels after drawing
        tick_labels = self.ax.get_xticklabels()
        if tick_labels and len(tick_labels) > 0:
            # Get the original text of the last label
            last_label_text = tick_labels[-1].get_text()
            
            # Only modify if ' km' is not already present
            if not last_label_text.endswith(' km'):
                # Create new labels list
                new_labels = [label.get_text() for label in tick_labels]
                new_labels[-1] = f'{last_label_text} km'
                
                # Use set_xticks and set_xticklabels together to avoid the warning
                ticks = self.ax.get_xticks()
                self.ax.set_xticks(ticks)
                self.ax.set_xticklabels(new_labels)

    def _append_km_to_last_xtick(self):
        """Append ' km' to the last xticklabel while preserving original formatting."""
        # Get current tick labels after drawing
        tick_labels = self.ax.get_xticklabels()
        if tick_labels and len(tick_labels) > 0:
            # Get the original text of the last label
            last_label_text = tick_labels[-1].get_text()
            
            # Create new labels list
            new_labels = [label.get_text() for label in tick_labels]
            new_labels[-1] = f'{last_label_text} km'
                
            # Use set_xticks and set_xticklabels together to avoid the warning
            ticks = self.ax.get_xticks()
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(new_labels)

    def __add_labels_to_xsection(self):
        self.set_horiz_extent()
        self.set_depth_extent()
        x1 = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03
        x2 = self.ax.get_xlim()[1] - (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03
        y = self.ax.get_ylim()[0] + (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.03

        self.ax.text(x1, y, "{}".format(self.properties["label"]),
                     verticalalignment='bottom', horizontalalignment='left',
                     path_effects=[pe.withStroke(linewidth=CROSSSECTION_DEFAULTS['text_stroke_linewidth'], foreground=CROSSSECTION_DEFAULTS['text_stroke_color'])])
        self.ax.text(x2, y, "{}'".format(self.properties["label"]),
                     verticalalignment='bottom', horizontalalignment='right',
                     path_effects=[pe.withStroke(linewidth=CROSSSECTION_DEFAULTS['text_stroke_linewidth'], foreground=CROSSSECTION_DEFAULTS['text_stroke_color'])])
        
        # Append ' km' to the last xticklabel
        self._append_km_to_last_xtick()

    def set_depth_extent(self, depth_extent=None):
        if depth_extent is None:
            self.ax.set_ylim(self.properties["depth_extent"])
        else:
            self.ax.set_ylim(depth_extent)

    def set_horiz_extent(self, extent=None):
        if extent is None:
            self.ax.set_xlim(0, self.properties["radius"]*2/1000)
        else:
            self.ax.set_xlim(extent)

    def plot(self, lat=None, lon=None, z=None, x=None, z_dir="depth", z_unit="m", **kwargs):
        if z is None:
            z = np.zeros_like(x if x is not None else lat)

        depth = np.asarray(z)

        if z_unit.lower() == "km":
            z_unit_conv = 1
        elif z_unit.lower() == "m":
            z_unit_conv = 1 / 1000
        else:
            raise ValueError(f"Invalid z_unit '{z_unit}'. Options: 'km' or 'm'.")

        if z_dir.lower() == "depth":
            z_dir_conv = -1
        elif z_dir.lower() == "elev":
            z_dir_conv = 1
        else:
            raise ValueError(f"Invalid z_dir '{z_dir}'. Options: 'depth' or 'elev'.")

        depth = depth * z_unit_conv * z_dir_conv

        if x is None:
            if lat is None or lon is None:
                raise ValueError("Either (lat, lon) or x must be provided.")
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")

        x = np.asarray(x)
        self.ax.plot(x, depth, **kwargs)

    def plot_catalog(self, catalog, s="magnitude", c="time", color=None, cmap="viridis_r", alpha=0.5, **kwargs):
        """
        Plot earthquake catalog on the cross-section.
        
        Creates a scatter plot of earthquake events from an ObsPy Catalog object
        projected onto the cross-section line, with customizable size, color, 
        and styling options.
        
        Parameters:
        -----------
        catalog : obspy.core.event.Catalog
            ObsPy Catalog object containing earthquake events
        s : str or array-like, optional
            Size parameter for scatter points. If "magnitude" (default), 
            point sizes are scaled by earthquake magnitude. Otherwise, 
            can be a numeric array or column name from catalog data.
        c : str or array-like, optional
            Color parameter for scatter points. If "time" (default), 
            points are colored by event time. Otherwise, can be a numeric 
            array, column name, or color specification.
        color : str or array-like, optional
            Alternative to 'c' parameter for specifying color. If provided,
            takes precedence over 'c'. Can be a single color name, hex code,
            or array of colors. Compatible with matplotlib's scatter color
            parameter.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for mapping numeric values to colors (default: "viridis_r")
        alpha : float, optional
            Transparency of scatter points, 0 (transparent) to 1 (opaque) 
            (default: 0.5)
        **kwargs
            Additional keyword arguments passed to matplotlib's scatter function
            
        Returns:
        --------
        matplotlib.collections.PathCollection
            The scatter plot collection
            
        Notes:
        ------
        - Point sizes are automatically scaled based on magnitude using the
          MagLegend class when s="magnitude"
        - Colors are mapped using the specified colormap when c or color 
          contains numeric data
        - Both 'c' and 'color' parameters are supported for compatibility
          with matplotlib's scatter function
        - Earthquake coordinates are automatically projected onto the 
          cross-section line using the project2line function
        - The cross-section extent is automatically adjusted after plotting
          
        Examples:
        ---------
        # Basic usage with default magnitude sizing and time coloring
        xs_obj.plot_catalog(catalog)
        
        # Custom size and color
        xs_obj.plot_catalog(catalog, s="magnitude", c="depth", cmap="plasma")
        
        # Use color parameter instead of c
        xs_obj.plot_catalog(catalog, color="red", alpha=0.8)
        
        # Custom size array and color mapping
        xs_obj.plot_catalog(catalog, s=[10, 20, 30], c=[1, 2, 3], cmap="cool")
        """
        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        # Handle both 'c' and 'color' parameters like matplotlib's scatter
        if color is not None:
            # 'color' parameter takes precedence over 'c'
            c = color
        elif c == "time":
            c = catdata["time"]
        else:
            c = c

        x = project2line(catdata["lat"], catdata["lon"], P1=self.A1, P2=self.A2, unit="km")
        scatter = self.ax.scatter(x, catdata["depth"], s=s, c=c, cmap=cmap, alpha=alpha, **kwargs)
        self.set_depth_extent()
        self.set_horiz_extent()
        return scatter

    def plot_inventory(self, inventory, marker_size=6, color='black', alpha=0.8, **kwargs):
        try:
            station_lats = []
            station_lons = []
            station_elevs = []
            
            for network in inventory:
                for station in network:
                    if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                        station_lats.append(station.latitude)
                        station_lons.append(station.longitude)
                        station_elevs.append(station.elevation)
            
            if station_lats and station_lons:
                x_coords = project2line(station_lats, station_lons, P1=self.A1, P2=self.A2, unit="km")
                elevs = np.full_like(x_coords, station_elevs)
                
                self.ax.scatter(x_coords, elevs/-1000,
                              s=marker_size, 
                              c=color, 
                              marker='v',
                              alpha=alpha,
                              **kwargs)
                self.set_depth_extent()
                self.set_horiz_extent()
            else:
                print("No valid station coordinates found in inventory for cross-section")
                
        except Exception as e:
            print(f"Error plotting inventory on cross-section: {e}")
            print("Continuing without cross-section inventory plot...")

    def plot_heatmap(self, *args, grid_size=HEATMAP_DEFAULTS['grid_size'], 
                     cmap=HEATMAP_DEFAULTS['cmap'], alpha=HEATMAP_DEFAULTS['alpha'], 
                     vmin=HEATMAP_DEFAULTS['vmin'], vmax=HEATMAP_DEFAULTS['vmax'], **kwargs):
        try:
            if len(args) == 1 and hasattr(args[0], 'events'):
                catalog = args[0]
                catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
                lat = catdata["lat"].values
                lon = catdata["lon"].values
                depth = catdata["depth"].values
            elif len(args) >= 2:
                lat = np.asarray(args[0])
                lon = np.asarray(args[1])
                depth = np.asarray(args[2]) if len(args) > 2 else None
            else:
                raise ValueError("Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, [depth], ...)")
            
            if len(lat) == 0 or len(lon) == 0:
                print("Warning: Empty coordinate arrays")
                return None
            
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")
            
            if len(x) == 0 or np.any(np.isnan(x)):
                print("Warning: Failed to project coordinates to cross-section line")
                return None
            
            depth_km = np.asarray(depth) / 1000.0
            depth_km = -depth_km
            
            x_min, x_max = np.min(x), np.max(x)
            depth_min, depth_max = np.min(depth_km), np.max(depth_km)
            
            grid_size_km = grid_size * 111.0
            
            if grid_size_km < 0.1:
                grid_size_km = 0.1
            
            data_range_x = x_max - x_min
            data_range_depth = depth_max - depth_min
            min_grid_size = max(0.1, min(data_range_x, data_range_depth) * 0.1)
            if grid_size_km < min_grid_size:
                grid_size_km = min_grid_size
            
            x_pad = (x_max - x_min) * 0.1
            depth_pad = (depth_max - depth_min) * 0.1
            
            if x_max <= x_min or depth_max <= depth_min:
                print("Warning: Invalid coordinate ranges for heatmap")
                return None
            
            x_grid = np.arange(x_min - x_pad, x_max + x_pad, grid_size_km)
            depth_grid = np.arange(depth_min - depth_pad, depth_max + depth_pad, grid_size_km)
            
            if len(x_grid) < 2 or len(depth_grid) < 2:
                print("Warning: Grid too small for heatmap")
                return None
            
            H, xedges, yedges = np.histogram2d(x, depth_km, bins=[x_grid, depth_grid])
            
            if H.size == 0 or np.all(H == 0):
                print("Warning: No data points in the specified region")
                return None
            
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            depth_centers = (yedges[:-1] + yedges[1:]) / 2
            x_mesh, depth_mesh = np.meshgrid(x_centers, depth_centers)
            
            im = self.ax.pcolormesh(x_mesh, depth_mesh, H.T, 
                                   cmap=cmap, alpha=alpha, 
                                   vmin=vmin, vmax=vmax,
                                   **kwargs)
            
            return im
            
        except Exception as e:
            print(f"Error creating cross-section heatmap: {e}")
            print("Continuing without heatmap...")
            return None


class TimeSeries:
    """
    TimeSeries class that creates time-series axes without inheriting from plt.Figure
    This avoids conflicts when used with SubFigures
    """

    name = "time-series"

    def __init__(self, fig=None, trange=None, axis_type="depth",
                 depth_extent=(-50., 4.), maglegend=MagLegend(),
                 colorbar=False, verbose=False, **kwargs):

        # Create figure if none provided
        if fig is None:
            # Extract figure-specific kwargs
            fig_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['dpi', 'figsize']}
            fig = plt.figure(**fig_kwargs)

        # Store the figure reference
        self.figure = fig
        
        # Remove any matplotlib figure-specific kwargs that would cause issues
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['dpi', 'figsize']}
        
        # Create the axes
        self.ax = fig.add_subplot(111, **plot_kwargs)
        
        # Set thicker spines
        for spine in self.ax.spines.values():
            spine.set_linewidth(AXES_DEFAULTS['spine_linewidth'])

        self.trange = trange
        self.depth_extent = depth_extent
        self.maglegend = maglegend
        self.orientation = "horizontal"
        self.axis_type = axis_type
        self.scatter_unit = "magnitude"

        # Colorbar axis setup
        if colorbar:
            self.axC = fig.add_axes([0.15, -0.1, 0.7, 0.05])
            # Apply centralized tick styling for main axis
            self.ax.tick_params(axis='both', 
                                labelcolor=TICK_DEFAULTS['labelcolor'],
                                labelsize=TICK_DEFAULTS['labelsize'],
                                color=TICK_DEFAULTS['tick_color'],
                                length=TICK_DEFAULTS['tick_size'],
                                width=TICK_DEFAULTS['tick_width'],
                                direction=TICK_DEFAULTS['tick_direction'],
                                pad=TICK_DEFAULTS['tick_pad'],
                                left=True, labelleft=True,
                                bottom=False, labelbottom=False,
                                right=False, labelright=False,
                                top=False, labeltop=False)
            # Apply centralized tick styling for colorbar axis
            self.axC.tick_params(labelcolor=TICK_DEFAULTS['labelcolor'],
                                 labelsize=TICK_DEFAULTS['labelsize'],
                                 color=TICK_DEFAULTS['tick_color'],
                                 length=TICK_DEFAULTS['tick_size'],
                                 width=TICK_DEFAULTS['tick_width'],
                                 direction=TICK_DEFAULTS['tick_direction'],
                                 pad=TICK_DEFAULTS['tick_pad'])
            self.axC.set_visible(True)
        else:
            # Apply centralized tick styling
            self.ax.tick_params(axis='both', 
                                labelcolor=TICK_DEFAULTS['labelcolor'],
                                labelsize=TICK_DEFAULTS['labelsize'],
                                color=TICK_DEFAULTS['tick_color'],
                                length=TICK_DEFAULTS['tick_size'],
                                width=TICK_DEFAULTS['tick_width'],
                                direction=TICK_DEFAULTS['tick_direction'],
                                pad=TICK_DEFAULTS['tick_pad'],
                                left=True, labelleft=True,
                                bottom=True, labelbottom=True,
                                right=False, labelright=False,
                                top=False, labeltop=False)

        # Set Y-Label using centralized styling
        if self.axis_type == "depth":
            ylabel = "Depth (km)"
        elif self.axis_type == "magnitude":
            ylabel = "Magnitude"
        self.set_ylim()
        self.ax.set_ylabel(ylabel, labelpad=CROSSSECTION_DEFAULTS['ylabel_pad'],
                           color=TICK_DEFAULTS['axes_labelcolor'], 
                           fontsize=TICK_DEFAULTS['axes_labelsize'])

        # Date formatting using ObsPy's datetime formatting
        if not colorbar:
            _set_xaxis_obspy_dates(self.ax)

    def set_ylim(self, ylim=None):
        if self.axis_type == "depth":
            if ylim is None:
                self.ax.set_ylim(self.depth_extent)
            else:
                self.ax.set_ylim(ylim)
        elif self.axis_type == "magnitude":
            if ylim is None:
                self.ax.set_ylim(self.mag_extent)
            else:
                self.ax.set_ylim(ylim)

    def scatter(self, t, y, yaxis="Depth", **kwargs):
        self.ax.scatter(convert_timeformat(t, "matplotlib"), y, **kwargs)
        self.set_ylim()

    def plot_catalog(self, catalog, s="magnitude", c="time", color=None, alpha=0.5, **kwargs):
        """
        Plot earthquake catalog on the time-series plot.
        
        Creates a scatter plot of earthquake events from an ObsPy Catalog object
        plotted against time, with customizable size, color, and styling options.
        The y-axis can represent either depth or magnitude based on the axis_type.
        
        Parameters:
        -----------
        catalog : obspy.core.event.Catalog
            ObsPy Catalog object containing earthquake events
        s : str or array-like, optional
            Size parameter for scatter points. If "magnitude" (default), 
            point sizes are scaled by earthquake magnitude. Otherwise, 
            can be a numeric array or column name from catalog data.
        c : str or array-like, optional
            Color parameter for scatter points. If "time" (default), 
            points are colored by event time. Otherwise, can be a numeric 
            array, column name, or color specification.
        color : str or array-like, optional
            Alternative to 'c' parameter for specifying color. If provided,
            takes precedence over 'c'. Can be a single color name, hex code,
            or array of colors. Compatible with matplotlib's scatter color
            parameter.
        alpha : float, optional
            Transparency of scatter points, 0 (transparent) to 1 (opaque) 
            (default: 0.5)
        **kwargs
            Additional keyword arguments passed to matplotlib's scatter function
            
        Returns:
        --------
        matplotlib.collections.PathCollection
            The scatter plot collection
            
        Notes:
        ------
        - Point sizes are automatically scaled based on magnitude using the
          MagLegend class when s="magnitude"
        - Colors are mapped using the specified colormap when c or color 
          contains numeric data
        - Both 'c' and 'color' parameters are supported for compatibility
          with matplotlib's scatter function
        - The y-axis represents either depth (when axis_type="depth") or 
          magnitude (when axis_type="magnitude")
        - The y-axis limits are automatically adjusted after plotting
          
        Examples:
        ---------
        # Basic usage with default magnitude sizing and time coloring
        ts_obj.plot_catalog(catalog)
        
        # Custom size and color for depth-time plot
        ts_obj.plot_catalog(catalog, s="magnitude", c="depth", alpha=0.8)
        
        # Use color parameter instead of c
        ts_obj.plot_catalog(catalog, color="red", alpha=0.8)
        
        # Custom size array and color mapping
        ts_obj.plot_catalog(catalog, s=[10, 20, 30], c=[1, 2, 3])
        """
        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        # Handle both 'c' and 'color' parameters like matplotlib's scatter
        if color is not None:
            # 'color' parameter takes precedence over 'c'
            c = color
        elif c == "time":
            c = catdata["time"]
        else:
            c = c

        if self.axis_type == "depth":
            scatter = self.ax.scatter(catdata["time"], catdata["depth"], s=s, c=c, alpha=alpha, **kwargs)
        if self.axis_type == "magnitude":
            scatter = self.ax.scatter(catdata["time"], catdata["mag"], s=s, c=c, alpha=alpha, **kwargs)
        self.set_ylim()
        return scatter

    def axvline(self, t, *args, **kwargs):
        self.ax.axvline(convert_timeformat(t, "matplotlib"), *args, **kwargs)


# Now update the VolcanoFigure class to use the new non-inheriting classes
class VolcanoFigure(plt.Figure):

    name = "volcano-figure"

    def __init__(self, *args, origin=(default_volcano["lat"], default_volcano["lon"]), radial_extent_km=50.0,
                 map_extent=None,  # [minlon, maxlon, minlat, maxlat] ([L, R, B, T])
                 ts_axis_type="depth",
                 depth_extent=(-50, 4),
                 xs1=None,
                 xs2=None,
                 figsize=(10, 10),  # Allow custom figure size
                 hillshade=False,
                 dpi=300,
                 **kwargs):

        # Set default DPI in kwargs if not already specified
        if 'dpi' not in kwargs:
            kwargs['dpi'] = dpi

        # Initialize the Figure object properly
        figscale = 8
        self.figscale = figscale
        
        # Create the main figure first
        super().__init__(figsize=figsize, **kwargs)

        # Define Map Properties
        self.properties = dict()
        self.properties["origin"] = origin
        self.properties["radial_extent_km"] = radial_extent_km
        self.properties["map_extent"] = radial_extent2map_extent(origin[0], origin[1], radial_extent_km)
        self.properties["depth_extent"] = depth_extent
        if map_extent:  # overwrite origin, radial_extent_km if map_extent is set explicitly
            self.properties["origin"] = None
            self.properties["radial_extent_km"] = None
            self.properties["map_extent"] = map_extent
        self.properties["ts_axis_type"] = ts_axis_type

        # Parse and prepare Cross-Section arguments
        xs1 = dict({}) if xs1 is None else xs1
        xs2 = dict({}) if xs2 is None else xs2
        xs1_defaults = dict({
            'origin': self.properties["origin"],
            'azimuth': 270,  # East-West
            'radius_km': self.properties["radial_extent_km"],
            'depth_extent': self.properties["depth_extent"],
            'label': 'A'
        })
        xs2_defaults = dict({
            'origin': self.properties["origin"],
            'azimuth': 0,  # North-South
            'radius_km': self.properties["radial_extent_km"],
            'depth_extent': self.properties["depth_extent"],
            'label': 'B'
        })

        ## Create subfigures
        spec = self.add_gridspec(1, 1)

        # subfigure - Map
        self.fig_m = self.add_subfigure(spec[0:1, 0:1])
        self.map_obj = Map(fig=self.fig_m, origin=origin, radial_extent_km=radial_extent_km)
        if hillshade:
            self.map_obj.add_hillshade()
        self.map_obj.add_scalebar()
        lbwh = np.array([0.9, 4.0, 3.0, 3.0])
        self.map_obj.ax.set_position(lbwh / figscale)

        # subfigure - Top Cross-Section
        self.fig_xs1 = self.add_subfigure(spec[0:1, 0:1])
        self.xs1_obj = CrossSection(fig=self.fig_xs1, **{**xs1_defaults, **xs1})  # overwrite defaults w user input
        lbwh = np.array([4.0, 5.8, 3.0, 1.2])
        self.xs1_obj.ax.set_position(lbwh / figscale)

        # subfigure - Bottom Cross-Section
        self.fig_xs2 = self.add_subfigure(spec[0:1, 0:1])
        self.xs2_obj = CrossSection(fig=self.fig_xs2, ** {**xs2_defaults, **xs2})  # overwrite defaults w user input
        lbwh = np.array([4.0, 4.0, 3.0, 1.2])
        self.xs2_obj.ax.set_position(lbwh / figscale)

        # subfigure - TimeSeries
        self.fig_ts = self.add_subfigure(spec[0:1, 0:1])
        self.ts_obj = TimeSeries(fig=self.fig_ts, depth_extent=self.properties["depth_extent"], axis_type=self.properties["ts_axis_type"])
        lbwh = np.array([0.9, 1.2, 6.1, 2.1])
        self.ts_obj.ax.set_position(lbwh / figscale)

        # subfigure - Legend
        self.fig_leg = self.add_subfigure(spec[0:1, 0:1])
        axL = self.fig_leg.add_subplot(111)
        lbwh = np.array([5.5, 1.2, 1.9, 2.1])
        axL.set_position(lbwh / figscale)
        axL.set_visible(False)

        # Plot Cross-Section lines to Map
        self.map_obj.plot_line(self.xs1_obj.properties["points"][0], self.xs1_obj.properties["points"][1], label=self.xs1_obj.properties["label"])
        self.map_obj.plot_line(self.xs2_obj.properties["points"][0], self.xs2_obj.properties["points"][1], label=self.xs2_obj.properties["label"])

    def info(self):
        print("::: VOLCANO FIGURE :::")
        print(self.properties)
        print()

    def add_hillshade(self, *args, **kwargs):
        self.map_obj.add_hillshade(*args, **kwargs)

    def add_ocean(self, *args, **kwargs):
        self.map_obj.add_ocean(*args, **kwargs)

    def add_coastline(self, *args, **kwargs):
        self.map_obj.add_coastline(*args, **kwargs)

    def add_terrain(self, *args, **kwargs):
        """Add terrain tiles from default source to map."""
        self.add_arcgis_terrain(*args, **kwargs)

    def add_arcgis_terrain(self, *args, **kwargs):
        """Add ArcGIS terrain tiles to the map."""
        self.map_obj.add_arcgis_terrain(*args, **kwargs)

    def add_google_terrain(self, *args, **kwargs):
        """Add Google terrain tiles to the map."""
        self.map_obj.add_google_terrain(*args, **kwargs)

    def add_google_street(self, *args, **kwargs):
        """Add Google street tiles to the map."""
        self.map_obj.add_google_street(*args, **kwargs)

    def add_google_satellite(self, *args, **kwargs):
        """Add Google satellite tiles to the map."""
        self.map_obj.add_google_satellite(*args, **kwargs)

    def plot(self, lat=None, lon=None, z=None, z_dir="depth", z_unit="m", transform=ccrs.Geodetic(), **kwargs):
        self.map_obj.plot(lat, lon, transform=transform, **kwargs)
        self.xs1_obj.plot(lat, lon, z=z, z_dir=z_dir, z_unit=z_unit, **kwargs)
        self.xs2_obj.plot(lat, lon, z=z, z_dir=z_dir, z_unit=z_unit, **kwargs)

    def scatter(self, lat=[], lon=[], x=[], time=[], y=[], transform=ccrs.Geodetic(), **kwargs):
        """WILL THIS WORK?"""
        self.map_obj.scatter(lat, lon, transform=transform, **kwargs)
        self.xs1_obj.scatter(lat, lon, **kwargs)
        self.xs2_obj.scatter(lat, lon, **kwargs)
        self.ts_obj.scatter(time, y, **kwargs)

    def plot_catalog(self, *args, transform=ccrs.Geodetic(), **kwargs):
        """
        Plot earthquake catalog on all subplots (map, cross-sections, and time-series).
        
        Creates scatter plots of earthquake events from an ObsPy Catalog object
        on the map, both cross-sections, and time-series plot with consistent
        styling across all subplots.
        
        Parameters:
        -----------
        catalog : obspy.core.event.Catalog
            ObsPy Catalog object containing earthquake events
        s : str or array-like, optional
            Size parameter for scatter points. If "magnitude" (default), 
            point sizes are scaled by earthquake magnitude.
        c : str or array-like, optional
            Color parameter for scatter points. If "time" (default), 
            points are colored by event time.
        color : str or array-like, optional
            Alternative to 'c' parameter for specifying color. If provided,
            takes precedence over 'c'.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for mapping numeric values to colors (default: "viridis_r")
        alpha : float, optional
            Transparency of scatter points (default: 0.5)
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for the map data (default: ccrs.Geodetic())
        **kwargs
            Additional keyword arguments passed to matplotlib's scatter function
            
        Returns:
        --------
        tuple
            Tuple containing the scatter plot collections from all subplots
            
        Notes:
        ------
        - All subplots use the same styling parameters for consistency
        - The map shows events in geographic coordinates
        - Cross-sections show events projected onto their respective lines
        - Time-series shows events plotted against time
        - Both 'c' and 'color' parameters are supported for compatibility
          with matplotlib's scatter function
        """
        map_scatter = self.map_obj.plot_catalog(*args, transform=transform, **kwargs)
        xs1_scatter = self.xs1_obj.plot_catalog(*args, **kwargs)
        xs2_scatter = self.xs2_obj.plot_catalog(*args, **kwargs)
        ts_scatter = self.ts_obj.plot_catalog(*args, **kwargs)
        return map_scatter, xs1_scatter, xs2_scatter, ts_scatter

    def plot_inventory(self, inventory, marker_size=8, color='black', alpha=0.8, 
                      transform=ccrs.Geodetic(), cross_section_marker_size=6, 
                      **kwargs):
        """
        Plot ObsPy inventory stations on the map and cross-sections.
        """
        # Plot inventory on the main map
        self.map_obj.plot_inventory(inventory, marker_size=marker_size, color=color, 
                                 alpha=alpha, transform=transform, **kwargs)
        
        # Plot inventory on both cross-sections
        self.xs1_obj.plot_inventory(inventory, marker_size=cross_section_marker_size, 
                                   color=color, alpha=alpha, **kwargs)
        self.xs2_obj.plot_inventory(inventory, marker_size=cross_section_marker_size, 
                                   color=color, alpha=alpha, **kwargs)

    def plot_heatmap(self, *args, **kwargs):
        """
        Plot a heatmap on the map and cross-sections.
        """
        # Plot heatmap on the main map using the stored map object
        self.map_obj.plot_heatmap(*args, **kwargs)
        
        # Plot heatmap on both cross-sections
        self.xs1_obj.plot_heatmap(*args, **kwargs)
        self.xs2_obj.plot_heatmap(*args, **kwargs)

    def title(self, t, x=0.5, y=0.975, **kwargs):
        self.suptitle(t, x=x, y=y, **kwargs)

    def subtitle(self, t, x=0.5, y=0.925, ha='center', va='center', **kwargs):
        self.text(x, y, t, ha=ha, va=va, **kwargs)

    def text(self, t, x=0.5, y=0.5, ha='center', va='center', **kwargs):
        super().text(x, y, t, ha=ha, va=va, **kwargs)

    def reftext(self, t, x=0.025, y=0.025, color="grey", ha="left", va="center", **kwargs):
        super().text(x, y, t, color=color, ha=ha, va=va, **kwargs)

    def catalog_subtitle(self, catalog):
        n = len(catalog)
        magnitudes = [event.magnitudes[0].mag for event in catalog if event.magnitudes]
        mmin = min(magnitudes)
        mmax = max(magnitudes)
        self.subtitle("{} Earthquakes | M{:2.1f}:M{:2.1f}".format(n, mmin, mmax))

    def magnitude_legend(self, cat):
        catdf = prep_catalog_data_mpl(cat)
        scatter = self.fig_leg.axes[0].scatter(catdf["lat"], catdf["lon"], s=catdf["size"], color="w", edgecolors="k")
        kw = dict(prop="sizes", num=3, fmt="M{x:1.0f}",
                  func=lambda s: np.sqrt(s / .3) / 3)
        legend = self.fig_leg.axes[0].legend(*scatter.legend_elements(**kw),
                                        loc="upper center", bbox_to_anchor=[0.0, 0.0, 1, 1],
                                        title="Magnitude", frameon=False)
        self.fig_leg.add_artist(legend)
        self.fig_leg.axes[0].set_visible(False)
