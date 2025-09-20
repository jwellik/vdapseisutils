"""
Python scripts for swarmmpl earthquake catalogs at volcanoes.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2024 February 28

TODO Address FutureWarning
/home/jwellik/miniconda3/envs/seismology312/lib/python3.12/site-packages/shapely/ops.py:276: FutureWarning: This function is deprecated. See: https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
  shell = type(geom.exterior)(zip(*func(*zip(*geom.exterior.coords))))

"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from vdapseisutils.utils.geoutils.geoutils import backazimuth
from vdapseisutils.utils.timeutils import convert_timeformat
from vdapseisutils.core.maps import elev_profile
from vdapseisutils.utils.geoutils import sight_point_pyproj, radial_extent2map_extent, project2line

from vdapseisutils.style import load_custom_rc
load_custom_rc("swarmmplrc")

# Plotting styles and formatters for maps and cross-sections
plt.rcParams['svg.fonttype'] = 'none'

# Use relative font sizing that automatically scales with figure dimensions
plt.rcParams['font.size'] = 8  # Base font size (small for crisp appearance)
plt.rcParams['axes.titlesize'] = 'large'      # 1.2x base = 9.6
plt.rcParams['axes.labelsize'] = 'medium'     # 1.0x base = 8
plt.rcParams['xtick.labelsize'] = 'small'     # 0.8x base = 6.4
plt.rcParams['ytick.labelsize'] = 'small'     # 0.8x base = 6.4
plt.rcParams['legend.fontsize'] = 'medium'    # 1.0x base = 8

# Set grey colors for labels and ticks
plt.rcParams['axes.labelcolor'] = 'grey'
plt.rcParams['xtick.color'] = 'grey'
plt.rcParams['ytick.color'] = 'grey'

# Keep legacy variables for backward compatibility (but they're no longer used)
titlefontsize = t1fs = plt.rcParams['axes.titlesize']
subtitlefontsize = t2fs = plt.rcParams['axes.labelsize']
axlabelfontsize = axlf = plt.rcParams['axes.labelsize']
annotationfontsize = afs = plt.rcParams['font.size']
axlabelcolor = axlc = 'grey'

# Heatmap defaults - change these values to update all plot_heatmap methods
HEATMAP_DEFAULTS = {
    'cmap': 'plasma',
    'alpha': 0.7,
    'grid_size': 0.01,  # Default to 0.05 degrees (≈5km) for VolcanoFigure
    'vmin': None,
    'vmax': None
}

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
        elif max_dimension > 5:   # Small local areas
            return "30s"
        elif max_dimension > 1:   # Very small local areas
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
def choose_scale_bar_length(map_width_km, fraction=0.3):
    """
    Given the width of the map in km, return the scale bar length (in km) as the value
    from ALLOWED_SCALES that is closest to fraction * map_width_km.
    """
    candidate = map_width_km * fraction
    # Choose the allowed scale that minimizes the absolute difference from candidate.
    ALLOWED_SCALES = [1, 5, 10, 20, 50, 100, 150, 200, 250, 500, 750, 1000, 5000, 10000]
    scale = min(ALLOWED_SCALES, key=lambda x: abs(x - candidate))
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
        
        # Set thicker spines (1.5x normal thickness) for visible spines
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.5)
        
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

##############################################################################################################
# Figure Classes


class Map(plt.Figure):

    name = "map"

    def __init__(self, *args, origin=(default_volcano["lat"], default_volcano["lon"]), radial_extent_km=50.0,
                 map_extent=None, # [minlon, maxlon, minlat, maxlat] ([L, R, B, T])
                 dpi=300, figsize=(8, 8), **kwargs):

        # Set default DPI in kwargs if not already specified
        if 'dpi' not in kwargs:
            kwargs['dpi'] = dpi

        # Initialize the Figure object properly
        fig = kwargs.pop('fig', None)
        if fig is None:
            # Create a new figure if none provided
            super().__init__(figsize=figsize, **kwargs)
        else:
            # Use the provided figure
            super().__init__(*args, **kwargs)
            # Copy only the necessary attributes, not the entire __dict__
            self._dpi = getattr(fig, '_dpi', dpi)
            self.canvas = fig.canvas
            # Don't set self.figure for SubFigure objects to avoid conflicts
            if not hasattr(fig, '_subfigure_spec'):
                self.figure = fig.figure if hasattr(fig, 'figure') else fig

        # Create the principal axes for the map
        # self.ax = self.add_subplot(*args, projection=ShadedReliefESRI().crs, **kwargs)  # creates principal GeoAxes
        
        # Remove dpi from kwargs since add_subplot doesn't accept it
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop('dpi', None)
        
        # Check if we're working with a SubFigure and handle accordingly
        if hasattr(fig, '_subfigure_spec'):
            # For SubFigure objects, create axes directly on the provided figure
            self.ax = fig.add_subplot(*args, projection=ccrs.Mercator(), **plot_kwargs)
        else:
            # For regular Figure objects, create axes on self
            self.ax = self.add_subplot(*args, projection=ccrs.Mercator(), **plot_kwargs)
        
        # Set thicker spines (1.5x normal thickness)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)

        # Define Map Properties
        self.properties = dict()
        self.properties["origin"] = origin
        self.properties["radial_extent_km"] = radial_extent_km
        self.properties["map_extent"] = radial_extent2map_extent(origin[0], origin[1], radial_extent_km)
        if map_extent:  # overwrite origin, radial_extent_km if map_extent is set explicitly
            self.properties["origin"] = None
            self.properties["radial_extent_km"] = None
            self.properties["map_extent"] = map_extent

        # # Create Axes, Set Extent, and Add Labels
        # super().__init__(*args, map_projection=ShadedReliefESRI().crs, **kwargs)
        self.ax.set_extent(self.properties["map_extent"])
        # self.ax.set_anchor("SW")  # Default is 'C' (But changing to SW has no effect?)
        # self.ax.set_xlabel("Map", fontsize=axlf, labelpad=5)  # This doesn't add anything, for some reason

        # Draw Grid and labels - grey lines and labels
        glv = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', alpha=0.5)
        glv.top_labels = False
        glv.bottom_labels = True
        glv.left_labels = True
        glv.right_labels = False
        glv.xlines = True
        # glv.ylines = True
        # Font sizes removed - uses rcParams automatically
        glv.xlabel_style = {'color': axlc}
        glv.ylabel_style = {'color': axlc}

        # # Draw lat,lon ticks and labels, no lines (Cartopy example)
        # from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        # glv = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, zorder=-10)
        # glv.bottom_labels = True
        # glv.left_labels = True
        # lon_formatter = LongitudeFormatter(zero_direction_label=True)
        # lat_formatter = LatitudeFormatter()
        # self.ax.xaxis.set_major_formatter(lon_formatter)
        # self.ax.yaxis.set_major_formatter(lat_formatter)


    def info(self):
        print("::: MAP AXES :::")
        print(self.properties)
        print()

    def add_hillshade(self, source="PyGMT", data_source="igpp", resolution="auto", 
                     topo=True, bath=False, radiance=[315, 60], alpha=0.8,
                     blend_mode="overlay", elevation_weight=0.3, hillshade_weight=0.7,
                     cmap="Greys_r", vertical_exag=1.5, normalize_elevation=True,
                     cache_data=True, **kwargs):
        """
        Add hillshade and elevation data to the map.
        
        Parameters:
        -----------
        source : str
            Data source ("PyGMT" only currently supported)
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
        alpha : float
            Overall transparency
        blend_mode : str
            Blending mode: "overlay", "multiply", "hillshade_only", "elevation_only"
        elevation_weight : float
            Weight for elevation data in blend (0-1)
        hillshade_weight : float
            Weight for hillshade data in blend (0-1)
        cmap : str
            Colormap for elevation data (default: "Greys_r" for black/white)
        vertical_exag : float
            Vertical exaggeration factor
        normalize_elevation : bool
            Normalize elevation data for better contrast
        cache_data : bool
            Cache downloaded data for reuse
        """
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

    def add_ocean(self, scale="10m", color="gray"):
        ocean = cfeature.NaturalEarthFeature(
            category='physical',
            name='ocean',
            scale=scale,
            facecolor=color),
        self.ax.add_feature(ocean[0])

    def add_coastline(self, scale="10m"):
        coastline = cfeature.COASTLINE.with_scale(scale),
        self.ax.add_feature(coastline[0])

    def add_locationmap(self, poi_lat, poi_lon, position=[0.68, 0.65, 0.3, 0.3]):
        spec = self.add_gridspec(8, 8)  # rows, columns (height, width)
        fig_loc = self.add_subfigure(spec[0:8, 0:8], zorder=2)  # rows from top, columns from left
        ax = fig_loc.add_axes(position, projection=ccrs.Orthographic(poi_lon, poi_lat))  # Orthographic(lon, lat)
        ax.add_feature(cfeature.OCEAN, zorder=0, color='white')
        ax.add_feature(cfeature.LAND, zorder=0, color='gray', edgecolor='black')
        ax.plot(poi_lon, poi_lat, "^r", transform=ccrs.Geodetic())
        ax.set_global()
        ax.gridlines()
        fig_loc.set_alpha(0.0)
        ax.set_alpha(0.0)

    def add_scalebar(self, scale_length_km="auto"):
        """ADD_SCALEBAR Uses Matplotlib's AnchoredSizeBar to make a scale bar in km
        https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar.html#

        Scale bar length will be determined automatically if a length is not provided.
        """

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm
        from vdapseisutils.utils.geoutils import backazimuth

        # Get scalebar length in axes percentage
        map_lonL, map_lonR, map_midlat = self.properties["map_extent"][0:3]
        az, d = backazimuth((map_midlat, map_lonL), (map_midlat, map_lonR))  # azimuth & map width (meters)
        map_width_km = d / 1000
        if scale_length_km == "auto":
            scale_length_km = choose_scale_bar_length(map_width_km, 0.30)  # auto determine scale bar km
        scale_bar_length_ax = scale_length_km / map_width_km

        # Add scale bar
        scalebar = AnchoredSizeBar(self.figure.axes[0].transAxes,  # Define length as % of axis width
                                   scale_bar_length_ax,  # Length in map units (degrees)
                                   f"{scale_length_km} km",  # Label
                                   'lower right',
                                   pad=0.5,
                                   color='black',
                                   frameon=False,
                                   size_vertical=0.01,  # Thickness of scale bar (as % of axis height)
                                   fontproperties=fm.FontProperties(size=10))

        self.figure.axes[0].add_artist(scalebar)

        print("Done.")


    def plot(self, lat, lon, *args, transform=ccrs.Geodetic(), **kwargs):
        self.ax.plot(lon, lat, *args, transform=transform, **kwargs)

    def scatter(self, lat, lon, size, color, transform=ccrs.Geodetic(), **kwargs):
        self.ax.scatter(lon, lat, size, color, transform=transform, **kwargs)

    def plot_catalog(self, catalog, s="magnitude", c="time", cmap="viridis_r", alpha=0.5, transform=ccrs.Geodetic(),
                     **kwargs):
        # from obspy import UTCDateTime
        # import matplotlib as mpl
        #
        # tmin = UTCDateTime(trange[0]) if trange is not None else catalog[0].origins[-1].time
        # tmax = UTCDateTime(trange[-1]) if trange is not None else catalog[0].origins[-1].time
        # norm = norm = mpl.colors.Normalize(vmin=tmin.matplotlib_date, vmax=tmax.matplotlib_date)

        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        if c == "time":
            c = catdata["time"]
        else:
            c = c

        self.ax.scatter(catdata["lon"], catdata["lat"], s=s, c=c, cmap=cmap, alpha=alpha, transform=transform, **kwargs)

    def plot_line(self, p1, p2, color="k", linewidth=1,
                  label=None, va='center', ha='center',
                  transform=ccrs.Geodetic(), **kwargs):
        label0 = label if label else ""  # Example: A
        label1 = label+"'" if label else ""  # Example: A'
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
        """
        Plot ObsPy inventory stations on the map.
        
        Parameters:
        -----------
        inventory : obspy.core.inventory.Inventory
            ObsPy inventory object containing station information
        marker_size : int or float
            Size of the station markers (default: 8, slightly smaller than normal)
        color : str
            Color of the station markers (default: 'black')
        alpha : float
            Transparency of the markers (default: 0.8)
        transform : cartopy.crs.Projection
            Coordinate reference system for plotting (default: Geodetic)
        **kwargs : additional arguments
            Additional arguments passed to ax.scatter()
        """
        try:
            # Extract station coordinates from inventory
            station_lats = []
            station_lons = []
            
            for network in inventory:
                for station in network:
                    # Get station coordinates
                    if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                        station_lats.append(station.latitude)
                        station_lons.append(station.longitude)
            
            if station_lats and station_lons:
                # Plot stations as upside-down triangles (v marker)
                self.ax.scatter(station_lons, station_lats, 
                              s=marker_size, 
                              c=color, 
                              marker='v',  # Upside-down triangle
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
        """
        Plot a heatmap on the map.
        
        Parameters:
        -----------
        *args : 
            Either: catalog (ObsPy Catalog object)
            Or: lat, lon, depth (depth is optional)
        grid_size : float
            Grid size in km for binning (default: 1.0)
        cmap : str
            Colormap for the heatmap (default: "plasma")
        alpha : float
            Transparency of the heatmap (default: 0.7)
        vmin, vmax : float, optional
            Minimum and maximum values for color scaling
        **kwargs : additional arguments
            Additional arguments passed to ax.pcolormesh()
        """
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
            
            # Check if we have valid data
            if len(lat) == 0 or len(lon) == 0:
                print("Warning: Empty coordinate arrays")
                return None
            
            # Create regular grid first
            lon_min, lon_max = np.min(lon), np.max(lon)
            lat_min, lat_max = np.min(lat), np.max(lat)
            
            # Convert grid_size from degrees to degrees (grid_size is already in degrees)
            grid_size_deg = grid_size
            
            # Ensure minimum grid size to avoid issues
            if grid_size_deg < 0.001:  # Minimum 0.001 degrees
                grid_size_deg = 0.001
            
            # Ensure grid size is not too small relative to data range
            data_range_lon = lon_max - lon_min
            data_range_lat = lat_max - lat_min
            min_grid_size = max(0.001, min(data_range_lon, data_range_lat) * 0.1)
            if grid_size_deg < min_grid_size:
                grid_size_deg = min_grid_size
            
            # Ensure we have valid ranges
            if lon_max <= lon_min or lat_max <= lat_min:
                print("Warning: Invalid coordinate ranges for heatmap")
                return None
            
            # Add some padding to the grid
            lon_pad = (lon_max - lon_min) * 0.1
            lat_pad = (lat_max - lat_min) * 0.1
            
            lon_grid = np.arange(lon_min - lon_pad, lon_max + lon_pad, grid_size_deg)
            lat_grid = np.arange(lat_min - lat_pad, lat_max + lat_pad, grid_size_deg)
            
            # Ensure grids have at least 2 points
            if len(lon_grid) < 2 or len(lat_grid) < 2:
                print("Warning: Grid too small for heatmap")
                return None
            
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(lon, lat, bins=[lon_grid, lat_grid])
            
            # Check if we have any data
            if H.size == 0 or np.all(H == 0):
                print("Warning: No data points in the specified region")
                return None
            
            # Get grid centers for plotting
            lon_centers = (xedges[:-1] + xedges[1:]) / 2
            lat_centers = (yedges[:-1] + yedges[1:]) / 2
            lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
            
            # Plot heatmap using pcolormesh
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


class CrossSection(plt.Figure):
    """
    CROSS-SECTION

    CrossSection is a Matplotlb Figure object with the plotting methods of the Axes object (e.g., plot(), scatter(),
    etc.) as well as custom plotting methods (e.g., plot_catalog(), add_profile(), etc.).

    Do everything in meters

    # DEPTHS and POSITIVE/NEGATIVE values
    The yaxis is set up such that depth in km is negative down (i.e., 50 km depth is a y value of -50)
    But the methods are set up to accept depth as positive down in m.
    In other words, passing a z value of +5000 will be automatically converted to a yaxis value of -50
    ? Should I change this ?
    """

    name = "cross-section"

    def __init__(self, *args,
                 points=[(46.198776, -122.261317), (46.197484, -122.122234)],  # W-E across MSH
                 origin=None,
                 radius_km=25.0,  # defaults to 50 km line
                 azimuth=270,  # defaults to W-E
                 map_extent=None,
                 depth_extent=(-50., 4.),
                 resolution=200.0,
                 max_n=100,
                 label="A",
                 width=None,  # width of cross-section box in km
                 maglegend=MagLegend(),
                 verbose=False,
                 figsize=(10, 6), dpi=300,
                 **kwargs):

        # Set default DPI in kwargs if not already specified
        if 'dpi' not in kwargs:
            kwargs['dpi'] = dpi

        # Initialize the figure properly
        fig = kwargs.pop('fig', None)
        if fig is None:
            # Create a new figure if none provided
            super().__init__(figsize=(8, 2), **kwargs)
        else:
            # Use the provided figure
            super().__init__(*args, **kwargs)
            # Copy only the necessary attributes, not the entire __dict__
            self._dpi = getattr(fig, '_dpi', dpi)
            self.canvas = fig.canvas
            # Don't set self.figure for SubFigure objects to avoid conflicts
            if not hasattr(fig, '_subfigure_spec'):
                self.figure = fig.figure if hasattr(fig, 'figure') else fig
        
        # Remove dpi from kwargs since add_subplot doesn't accept it
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop('dpi', None)
        
        # Check if we're working with a SubFigure and handle accordingly
        if hasattr(fig, '_subfigure_spec'):
            # For SubFigure objects, create axes directly on the provided figure
            self.ax = fig.add_subplot(**plot_kwargs)
        else:
            # For regular Figure objects, create axes on self
            self.ax = self.add_subplot(**plot_kwargs)
        
        # Set thicker spines (1.5x normal thickness)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)

        # Better organization?
        self.properties = dict()

        # Determine A & A' locations, etc.
        # If origin is given,
        # determine the points based on an origin, azimuth and distance
        if origin:
            self.properties["origin"] = origin
            self.properties["azimuth"] = azimuth
            self.properties["radius"] = radius_km * 1000
            self.properties["points"] = [np.nan, np.nan]
            self.properties["points"][0] = sight_point_pyproj(origin, azimuth,
                                                              self.properties["radius"])  # distance reqd in km
            self.properties["points"][1] = sight_point_pyproj(origin, np.mod(azimuth + 180, 360),
                                                              self.properties["radius"])
        else:
            if len(points) != 2:
                raise ValueError("ERROR: Points must be a list of 2 tuples of lat,lon coordinates.")
            self.properties["points"] = points
            self.properties["origin"] = None
            self.properties["azimuth"], self.properties["length"] = backazimuth(points[0], points[1])  # Returns length in km; does this get used?
            self.properties["radius"] = None

        # Extents by which to filter input data
        self.properties["map_extent"] = map_extent
        self.properties["depth_extent"] = depth_extent
        self.properties["depth_range"] = depth_extent[1] - depth_extent[0]  # height of cross section (km)

        self.properties["label"] = label  # Name of cross-section (eg, A)
        self.properties["full_label"] = "{a}-{a}'".format(a=self.properties["label"])  # eg, A-A'
        self.properties["width"] = width  # Width of cross-section box in km

        self.properties["orientation"] = "horizontal"  # horizontal or vertical layout

        self.verbose = verbose

        # Stub (I probably shouldn't redefine these here)
        self.A1 = self.properties["points"][0]
        self.A2 = self.properties["points"][1]

        self.profile = elev_profile.TopographicProfile([self.A1, self.A2], resolution=resolution, max_n=max_n)
        # if np.any(self.profile.get_elevation):
        if np.any(self.profile.elevation):
            self.__add_profile()

        self.__setup_axis_formatting()
        self.__add_labels_to_xsection()

    def set_depth_extent(self, depth_extent=None):
        if depth_extent is None:
            self.ax.set_ylim(self.properties["depth_extent"])
        else:
            self.ax.set_ylim(depth_extent)

    def set_horiz_extent(self, extent=None):
        """
        Specifying extent is 0,+rad_km*2 (distance along the line)

        :param extent:
        :return:
        """

        if extent is None:
            # self.ax.set_xlim(-self.properties["radius"]/1000, self.properties["radius"]/1000)  # centered distance
            self.ax.set_xlim(0, self.properties["radius"]*2/1000)  # distance along line
        else:
            self.ax.set_xlim(extent)

    def __add_profile(self):

        # Plot data and format axis for A-A'
        hd = self.profile.distance / 1000  # horizontal distance along line (convert meters to km)
        # elev = np.array(self.profile.get_elevation / 1000)  # elevation (convert m to km)
        elev = np.array(self.profile.elevation / 1000)  # elevation (convert m to km)
        # self.ax.set_ylim(self.properties["depth_extent"])
        self.plot(x=hd, z=elev, z_dir="elev", color="k", z_unit="km",
                  linewidth=0.75)  # Use CrossSection.plot() instead of Axes.plot()
        self.set_depth_extent()
        self.set_horiz_extent()

        # custom spine bounds for a nice clean look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines["left"].set_bounds(
            (self.properties["depth_extent"][0], elev[0]))  # depth_extent_v[1] is the top elev
        self.ax.spines["right"].set_bounds(self.properties["depth_extent"][0], elev[-1])

    def __setup_axis_formatting(self):
        self.ax.tick_params(axis='both', labelcolor=axlc,
                            left=False, labelleft=False,
                            bottom=True, labelbottom=True,
                            right=True, labelright=True,
                            top=False, labeltop=False)
        self.ax.yaxis.set_label_position("right")
        self.ax.set_ylabel("Depth (km)", rotation=270, labelpad=15)  # fontsize removed - uses rcParams
        self.ax.set_xlabel("Distance (km)", labelpad=10)  # fontsize removed - uses rcParams

    def __add_labels_to_xsection(self):
        # Add labels on cross-section using data coordinates for proper positioning
        self.set_horiz_extent()  # Ensure that horizontal axis limits are set
        self.set_depth_extent()  # Ensure that vertical axis limits are set
        x1 = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03  # 5% from left edge
        x2 = self.ax.get_xlim()[1] - (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03  # 5% from right edge
        y = self.ax.get_ylim()[0] + (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.03  # 5% from bottom

        self.ax.text(x1, y, "{}".format(self.properties["label"]),
                     verticalalignment='bottom', horizontalalignment='left',
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        self.ax.text(x2, y, "{}'".format(self.properties["label"]),
                     verticalalignment='bottom', horizontalalignment='right',
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def get_ve(self):  # return vertical exageration
        # return (plot_height/self.depth_range) / (plot_width/self.hd)  # vertical exaggeration = vertical scale / horizontal scale
        print("GET_VE not yet implemented :-(")

    def plot_to_map(self, ax=None):

        """
        Plot the cross-section line on a map.

        Parameters
        ----------
        ax : GeoAxes, optional
            The map axes to plot on. If None, creates new figure.

        Returns
        -------
        ax : GeoAxes
            The map axes with the cross-section line plotted
        """
        print("WARNING: THIS METHOD IS UNVERIFIED.")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=ccrs.Mercator())

        # Plot the main line
        ax.plot([self.A1[1], self.A2[1]], [self.A1[0], self.A2[0]],
                color='k', linewidth=1, transform=ccrs.Geodetic())

        # Add labels at endpoints
        ax.text(self.A1[1], self.A1[0], f"{self.properties['label']}",
                transform=ccrs.Geodetic(),
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))
        ax.text(self.A2[1], self.A2[0], f"{self.properties['label']}'",
                transform=ccrs.Geodetic(),
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))

        # If width is specified, draw the box
        if self.properties["width"] is not None:
            from pyproj import Geod
            g = Geod(ellps='WGS84')

            # Calculate perpendicular points
            az, baz, dist = g.inv(self.A1[1], self.A1[0], self.A2[1], self.A2[0])

            # Get perpendicular azimuths
            perp_az1 = az + 90
            perp_az2 = az - 90

            # Calculate corner points for first end
            lon1a, lat1a, _ = g.fwd(self.A1[1], self.A1[0], perp_az1, self.properties["width"] * 500)
            lon1b, lat1b, _ = g.fwd(self.A1[1], self.A1[0], perp_az2, self.properties["width"] * 500)

            # Calculate corner points for second end
            lon2a, lat2a, _ = g.fwd(self.A2[1], self.A2[0], perp_az1, self.properties["width"] * 500)
            lon2b, lat2b, _ = g.fwd(self.A2[1], self.A2[0], perp_az2, self.properties["width"] * 500)

            # Plot the box
            ax.plot([lon1a, lon2a], [lat1a, lat2a], 'k--', linewidth=1, transform=ccrs.Geodetic())
            ax.plot([lon1b, lon2b], [lat1b, lat2b], 'k--', linewidth=1, transform=ccrs.Geodetic())

        return ax

    def plot_catalog(self, catalog, s="magnitude", c="time", cmap="viridis_r", alpha=0.5, **kwargs):
    # from obspy import UTCDateTime
        # import matplotlib as mpl
        #
        # tmin = UTCDateTime(trange[0]) if trange is not None else catalog[0].origins[-1].time
        # tmax = UTCDateTime(trange[-1]) if trange is not None else catalog[0].origins[-1].time
        # norm = norm = mpl.colors.Normalize(vmin=tmin.matplotlib_date, vmax=tmax.matplotlib_date)

        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        if c == "time":
            c = catdata["time"]
        else:
            c = c

        x = project2line(catdata["lat"], catdata["lon"], P1=self.A1, P2=self.A2, unit="km")
        self.ax.scatter(x, catdata["depth"], s=s, c=c, cmap=cmap, alpha=alpha, **kwargs)
        self.set_depth_extent()  # Ensure depth extent remains the same
        self.set_horiz_extent()

    def plot_inventory(self, inventory, marker_size=6, color='black', alpha=0.8, **kwargs):
        """
        Plot ObsPy inventory stations on the cross-section.
        
        Parameters:
        -----------
        inventory : obspy.core.inventory.Inventory
            ObsPy inventory object containing station information
        marker_size : int or float
            Size of the station markers (default: 6, smaller than map view)
        color : str
            Color of the station markers (default: 'black')
        alpha : float
            Transparency of the markers (default: 0.8)
        **kwargs : additional arguments
            Additional arguments passed to ax.scatter()
        """
        try:
            # Extract station coordinates from inventory
            station_lats = []
            station_lons = []
            station_elevs = []
            
            for network in inventory:
                for station in network:
                    # Get station coordinates
                    if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                        station_lats.append(station.latitude)
                        station_lons.append(station.longitude)
                        station_elevs.append(station.elevation)  # meters
            
            if station_lats and station_lons:
                # Project station coordinates to the cross-section line
                x_coords = project2line(station_lats, station_lons, P1=self.A1, P2=self.A2, unit="km")
                
                # Set depth for all stations (at surface with optional offset)
                elevs = np.full_like(x_coords, station_elevs)
                
                # Plot stations as upside-down triangles
                self.ax.scatter(x_coords, elevs/-1000,  # convert elevation km to meters
                              s=marker_size, 
                              c=color, 
                              marker='v',  # Upside-down triangle
                              alpha=alpha,
                              **kwargs)
                self.set_depth_extent()  # Ensure depth extent remains the same
                self.set_horiz_extent()
            else:
                print("No valid station coordinates found in inventory for cross-section")
                
        except Exception as e:
            print(f"Error plotting inventory on cross-section: {e}")
            print("Continuing without cross-section inventory plot...")

    # def plot(self, lat=[], lon=[], z=[], x=[], z_dir="depth", z_unit="m", *args, **kwargs):
    #     # Points can be given as lat, lon, z, ...
    #     # OR as x, z
    #     # If distance is given as x, lat,lon are ignored
    #     # Assumes z is given as meters depth (down)
    #     # Creates a negative axis
    #     # E.g., 5000 m depth is plotted as -5
    #     # E.g., 3200 m elevation is plotted as 3.2
    #     # NOTE: This function is used to make the topographic line
    #
    #     if not np.any(z):
    #          raise ValueError("ERROR: Depth must be provided.")
    #
    #     if z_unit.lower() == "km":
    #         z_unit_conv = 1
    #     elif z_unit.lower() == "m":
    #         z_unit_conv = 1 / 1000  # convert m to km
    #     else:
    #         raise Warning("Z_UNIT {} not undertsood. Options are 'km' or 'm'. Using km.".format(z_unit))
    #
    #     if z_dir.lower() == "depth":
    #         z_dir_conv = -1  # Depths should be plotted as negative
    #     elif z_dir.lower() == "elev":
    #         z_dir_conv = 1
    #     else:
    #         raise Warning("Z_DIR {} not undertsood. Options are 'depth' or 'elev'. Using depth.".format(z_unit))
    #
    #     z = np.array(z) * z_unit_conv * z_dir_conv  # Convert to km and make negative for swarmmpl purposes
    #     if not np.any(x):
    #         x = project2line(lat, lon, P1=self.A1, P2=self.A2)  # returned as array
    #     self.ax.plot(x, z, *args, **kwargs)

    def plot(self, lat=None, lon=None, z=None, x=None, z_dir="depth", z_unit="m", **kwargs):
        """
        Plot data as a cross-section.

        Parameters:
        - lat, lon, depth: Arrays of latitude, longitude, and depth (optional)
        - x, depth: If x is given, lat/lon are ignored
        - z_dir: "depth" (default, plotted as negative) or "elev" (positive)
        - z_unit: "m" (default) or "km" (for conversion)
        - **kwargs: Passed to self.ax.plot()

        If depth is not provided, it defaults to 0.
        """

        # Ensure depth is an array and default to 0 if not provided
        if z is None:
            z = np.zeros_like(x if x is not None else lat)

        depth = np.asarray(z)

        # Handle unit conversion
        if z_unit.lower() == "km":
            z_unit_conv = 1
        elif z_unit.lower() == "m":
            z_unit_conv = 1 / 1000  # Convert meters to kilometers
        else:
            raise ValueError(f"Invalid z_unit '{z_unit}'. Options: 'km' or 'm'.")

        # Handle depth/elevation direction
        if z_dir.lower() == "depth":
            z_dir_conv = -1  # Depths should be negative
        elif z_dir.lower() == "elev":
            z_dir_conv = 1
        else:
            raise ValueError(f"Invalid z_dir '{z_dir}'. Options: 'depth' or 'elev'.")

        # Convert depth values
        depth = depth * z_unit_conv * z_dir_conv  # Convert to km, apply depth convention

        # Handle x values: if not given, compute from lat/lon
        if x is None:
            if lat is None or lon is None:
                raise ValueError("Either (lat, lon) or x must be provided.")
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")  # Compute projected x-coordinates

        # Ensure x is an array
        x = np.asarray(x)

        self.ax.plot(x, depth, **kwargs)

    def plot_point(self, x, z, *args, **kwargs):
        self.ax.plot(x, z, *args, **kwargs)

    def scatter(self, lat=[], lon=[], z=[], x=[], z_dir="depth", z_unit="m", *args, **kwargs):
        # Assumes z is given as meters depth (down)
        # lat,lon coordinates can be given, or x (distance along axis) can be given
        # if x is given, lat,lon are ignored
        # Creates a negative axis
        # E.g., 5000 m depth is plotted as -5
        # E.g., 3200 m elevation is plotted as 3.2

        z_unit_conv = 1000 if z_unit == "m" else 1
        z_dir_conv = -1 if z_dir == "depth" else 1

        z = np.array(z) / z_unit_conv * z_dir_conv  # Convert to km and make negative for swarmmpl purposes

        if not x:
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")  # Put lat, lon into a list or not?

        self.ax.scatter(x, z, *args, **kwargs)

    def info(self):
        print("::: Cross-Section ({}) :::".format(self.properties["label"]))
        # print("    - P1-P2 : (lat,lon)-(lat,lon) (... km @ ... deg) ")
        # print("    - depth_extent : (lat,lon)")
        # print("    - # EQs : ")
        print()

    def plot_heatmap(self, *args, grid_size=HEATMAP_DEFAULTS['grid_size'], 
                     cmap=HEATMAP_DEFAULTS['cmap'], alpha=HEATMAP_DEFAULTS['alpha'], 
                     vmin=HEATMAP_DEFAULTS['vmin'], vmax=HEATMAP_DEFAULTS['vmax'], **kwargs):
        """
        Plot a heatmap on the cross-section.
        
        Parameters:
        -----------
        *args : 
            Either: catalog (ObsPy Catalog object)
            Or: lat, lon, depth (depth is optional)
        grid_size : float
            Grid size in km for binning (default: 1.0)
        cmap : str
            Colormap for the heatmap (default: "plasma")
        alpha : float
            Transparency of the heatmap (default: 0.7)
        vmin, vmax : float, optional
            Minimum and maximum values for color scaling
        **kwargs : additional arguments
            Additional arguments passed to ax.pcolormesh()
        """
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
            
            # Check if we have valid data
            if len(lat) == 0 or len(lon) == 0:
                print("Warning: Empty coordinate arrays")
                return None
            
            # Project lat/lon coordinates to the cross-section line
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")
            
            # Check if projection was successful
            if len(x) == 0 or np.any(np.isnan(x)):
                print("Warning: Failed to project coordinates to cross-section line")
                return None
            
            # Convert depth to km and make negative for plotting
            depth_km = np.asarray(depth) / 1000.0  # Convert m to km
            depth_km = -depth_km  # Make negative for plotting
            
            # Create regular grid for x and depth first
            x_min, x_max = np.min(x), np.max(x)
            depth_min, depth_max = np.min(depth_km), np.max(depth_km)
            
            # Convert grid_size from degrees to km (1 degree ≈ 111 km)
            grid_size_km = grid_size * 111.0
            
            # Ensure minimum grid size to avoid issues
            if grid_size_km < 0.1:  # Minimum 0.1 km
                grid_size_km = 0.1
            
            # Ensure grid size is not too small relative to data range
            data_range_x = x_max - x_min
            data_range_depth = depth_max - depth_min
            min_grid_size = max(0.1, min(data_range_x, data_range_depth) * 0.1)
            if grid_size_km < min_grid_size:
                grid_size_km = min_grid_size
            
            # Add some padding to the grid
            x_pad = (x_max - x_min) * 0.1
            depth_pad = (depth_max - depth_min) * 0.1
            
            # Ensure we have valid ranges
            if x_max <= x_min or depth_max <= depth_min:
                print("Warning: Invalid coordinate ranges for heatmap")
                return None
            
            x_grid = np.arange(x_min - x_pad, x_max + x_pad, grid_size_km)
            depth_grid = np.arange(depth_min - depth_pad, depth_max + depth_pad, grid_size_km)
            
            # Ensure grids have at least 2 points
            if len(x_grid) < 2 or len(depth_grid) < 2:
                print("Warning: Grid too small for heatmap")
                return None
            
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(x, depth_km, bins=[x_grid, depth_grid])
            
            # Check if we have any data
            if H.size == 0 or np.all(H == 0):
                print("Warning: No data points in the specified region")
                return None
            
            # Get grid centers for plotting
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            depth_centers = (yedges[:-1] + yedges[1:]) / 2
            x_mesh, depth_mesh = np.meshgrid(x_centers, depth_centers)
            
            # Plot heatmap using pcolormesh
            im = self.ax.pcolormesh(x_mesh, depth_mesh, H.T, 
                                   cmap=cmap, alpha=alpha, 
                                   vmin=vmin, vmax=vmax,
                                   **kwargs)
            
            return im
            
        except Exception as e:
            print(f"Error creating cross-section heatmap: {e}")
            print("Continuing without heatmap...")
            return None


class TimeSeries(plt.Figure):
    """
    Creates time-series plot for geophysical data

    * Depths below sea-level are negative
    """

    name = "time-series"

    def __init__(self, *args,
                 trange=None,
                 axis_type="depth",
                 depth_extent=(-50., 4.),
                 maglegend=MagLegend(),
                 colorbar=False,
                 verbose=False,
                 dpi=300,
                 **kwargs):

        # Set default DPI in kwargs if not already specified
        if 'dpi' not in kwargs:
            kwargs['dpi'] = dpi

        # Initialize Figure properly
        fig = kwargs.pop('fig', None)
        if fig is None:
            # Create a new figure if none provided
            super().__init__(figsize=(8, 2), **kwargs)
        else:
            # Use the provided figure
            super().__init__(*args, **kwargs)
            # Copy only the necessary attributes, not the entire __dict__
            self._dpi = getattr(fig, '_dpi', dpi)
            self.canvas = fig.canvas
            # Don't set self.figure for SubFigure objects to avoid conflicts
            if not hasattr(fig, '_subfigure_spec'):
                self.figure = fig.figure if hasattr(fig, 'figure') else fig
        
        # Remove dpi from kwargs since add_subplot doesn't accept it
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop('dpi', None)
        
        # Check if we're working with a SubFigure and handle accordingly
        if hasattr(fig, '_subfigure_spec'):
            # For SubFigure objects, create axes directly on the provided figure
            self.ax = fig.add_subplot(**plot_kwargs)
        else:
            # For regular Figure objects, create axes on self
            self.ax = self.add_subplot(**plot_kwargs)
        
        # Set thicker spines (1.5x normal thickness)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)

        self.trange = trange
        self.depth_extent = depth_extent
        self.maglegend = maglegend
        self.orientation = "horizontal"  # horizontal or vertical layout
        self.axis_type = axis_type
        self.scatter_unit = "magnitude"  # not yet implemented

        # Colorbar axis setup
        if colorbar:
            # Create colorbar axis if colorbar is requested
            self.axC = self.add_axes([0.15, -0.1, 0.7, 0.05])  # [left, bottom, width, height]
            self.ax.tick_params(axis='both', labelcolor=axlc,
                                left=True, labelleft=True,
                                bottom=False, labelbottom=False,
                                right=False, labelright=False,
                                top=False, labeltop=False)
            self.axC.tick_params(labelcolor=axlc)
            self.axC.set_visible(True)
        else:
            self.ax.tick_params(axis='both', labelcolor=axlc,
                                left=True, labelleft=True,
                                bottom=True, labelbottom=True,
                                right=False, labelright=False,
                                top=False, labeltop=False)

        # Set Y-Label
        if self.axis_type == "depth":
            ylabel = "Depth (km)"
        elif self.axis_type == "magnitude":
            ylabel = "Magnitude"
        self.set_ylim()
        self.ax.set_ylabel(ylabel, labelpad=15)  # fontsize removed - uses rcParams

        # norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)  # (as mpldate)
        loc = mdates.AutoDateLocator()  # from matplotlib import dates as mdates
        formatter = mdates.ConciseDateFormatter(loc, show_offset=True)
        if colorbar:
            pass
            # cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            #                   cax=axC, orientation='horizontal', label='Time')
            # cb.ax.xaxis.set_major_locator(loc)
            # cb.ax.xaxis.set_major_formatter(formatter)
        else:
            self.ax.xaxis.set_major_locator(loc)
            self.ax.xaxis.set_major_formatter(formatter)
        # self.set_xlim([tmin, tmax])  # Set time extent of time series axis (as mpldate)

    def set_ylim(self, ylim=None):
        print(self.depth_extent)
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

    def plot_catalog(self, catalog, s="magnitude", c="time", alpha=0.5, **kwargs):

        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s
        if c == "time":
            c = catdata["time"]
        else:
            c = c

        if self.axis_type == "depth":
            self.ax.scatter(catdata["time"], catdata["depth"], s=s, c=c, alpha=alpha, **kwargs)
        if self.axis_type == "magnitude":
            self.ax.scatter(catdata["time"], catdata["mag"], s=s, c=c, alpha=alpha, **kwargs)
        self.set_ylim()

    def axvline(self, t, *args, **kwargs):
        self.ax.axvline(convert_timeformat(t, "matplotlib"), *args, **kwargs)


##############################################################################################################
# Combo Packages


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
        
        # Store the figure reference for subfigure creation
        main_fig = self

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
        spec = main_fig.add_gridspec(1, 1)

        # subfigure - Map
        self.fig_m = main_fig.add_subfigure(spec[0:1, 0:1])
        self.map_obj = Map(fig=self.fig_m, origin=origin, radial_extent_km=radial_extent_km)
        # Copy the map's ax to the subfigure
        self.fig_m.ax = self.map_obj.ax
        if hillshade:
            self.map_obj.add_hillshade()
        self.map_obj.add_scalebar()
        lbwh = np.array([0.9, 4.0, 3.0, 3.0])
        self.fig_m.ax.set_position(lbwh / figscale)

        # subfigure - Top Cross-Section
        self.fig_xs1 = main_fig.add_subfigure(spec[0:1, 0:1])
        # self.fig_xs1 = CrossSection(fig=self.fig_xs1, origin=origin, radius_km=radial_extent_km,
        #                        depth_extent=depth_extent, azimuth=270, label="A")
        xs1_obj = CrossSection(fig=self.fig_xs1, **{**xs1_defaults, **xs1})  # overwrite defaults w user input
        # Copy the cross-section's ax to the subfigure
        self.fig_xs1.ax = xs1_obj.ax
        lbwh = np.array([4.0, 5.8, 3.0, 1.2])
        self.fig_xs1.ax.set_position(lbwh / figscale)

        # subfigure - Bottom Cross-Section
        self.fig_xs2 = main_fig.add_subfigure(spec[0:1, 0:1])
        # self.fig_xs2 = CrossSection(fig=self.fig_xs2,  origin=origin, radius_km=radial_extent_km,
        #                        depth_extent=depth_extent, azimuth=0, label="B")
        xs2_obj = CrossSection(fig=self.fig_xs2, ** {**xs2_defaults, **xs2})  # overwrite defaults w user input
        # Copy the cross-section's ax to the subfigure
        self.fig_xs2.ax = xs2_obj.ax
        lbwh = np.array([4.0, 4.0, 3.0, 1.2])
        self.fig_xs2.ax.set_position(lbwh / figscale)

        # subfigure - TimeSeries
        self.fig_ts = main_fig.add_subfigure(spec[0:1, 0:1])
        ts_obj = TimeSeries(fig=self.fig_ts, depth_extent=self.properties["depth_extent"], axis_type=self.properties["ts_axis_type"])
        # Copy the time-series's ax to the subfigure
        self.fig_ts.ax = ts_obj.ax
        lbwh = np.array([0.9, 1.2, 6.1, 2.1])
        self.fig_ts.ax.set_position(lbwh / figscale)

        # subfigure - Legend
        self.fig_leg = main_fig.add_subfigure(spec[0:1, 0:1])
        axL = self.fig_leg.add_subplot(111)
        lbwh = np.array([5.5, 1.2, 1.9, 2.1])
        axL.set_position(lbwh / figscale)
        axL.set_visible(False)

        # Plot Cross-Section lines to Map
        self.map_obj.plot_line(self.fig_xs1.properties["points"][0], self.fig_xs1.properties["points"][1], label=self.fig_xs1.properties["label"])
        self.map_obj.plot_line(self.fig_xs2.properties["points"][0], self.fig_xs2.properties["points"][1], label=self.fig_xs2.properties["label"])


    def info(self):
        print("::: VOLCANO FIGURE :::")
        print(self.properties)
        print()

    def add_hillshade(self, *args, **kwargs):
        self.fig_m.add_hillshade(*args, **kwargs)

    def add_ocean(self, *args, **kwargs):
        self.fig_m.add_ocean(*args, **kwargs)

    def add_coastline(self, *args, **kwargs):
        self.fig_m.add_ocean(*args, **kwargs)

    def plot(self, lat=None, lon=None, z=None, z_dir="depth", z_unit="m", transform=ccrs.Geodetic(), **kwargs):
        self.fig_m.plot(lat, lon, transform=transform, **kwargs)
        self.fig_xs1.plot(lat, lon, z=z, z_dir=z_dir, z_unit=z_unit, **kwargs)
        self.fig_xs2.plot(lat, lon, z=z, z_dir=z_dir, z_unit=z_unit, **kwargs)
        # self.fig_xs1.plot(x, y, 0, **kwargs)
        # self.fig_xs2.plot(x, y, 0, **kwargs)
        # self.fig_ts.plot(time, y, **kwargs)

    def scatter(self, lat=[], lon=[], x=[], time=[], y=[], transform=ccrs.Geodetic(), **kwargs):
        """WILL THIS WORK?"""
        self.fig_m.scatter(lat, lon, transform=transform)
        self.fig_xs1.scatter(lat, lon, **kwargs)
        self.fig_xs2.scatter(lat, lon, **kwargs)
        self.fig_xs1.scatter(x, y, **kwargs)
        self.fig_xs2.scatter(x, y, **kwargs)
        self.fig_ts.scatter(time, y, **kwargs)

    def plot_catalog(self, *args, transform=ccrs.Geodetic(), **kwargs):
        self.map_obj.plot_catalog(*args, transform=transform, **kwargs)
        self.fig_xs1.plot_catalog(*args, **kwargs)
        self.fig_xs2.plot_catalog(*args, **kwargs)
        self.fig_ts.plot_catalog(*args, **kwargs)

    def plot_inventory(self, inventory, marker_size=8, color='black', alpha=0.8, 
                      transform=ccrs.Geodetic(), cross_section_marker_size=6, 
                      **kwargs):
        """
        Plot ObsPy inventory stations on the map and cross-sections.
        
        Parameters:
        -----------
        inventory : obspy.core.inventory.Inventory
            ObsPy inventory object containing station information
        marker_size : int or float
            Size of the station markers on the main map (default: 8)
        color : str
            Color of the station markers (default: 'black')
        alpha : float
            Transparency of the markers (default: 0.8)
        transform : cartopy.crs.Projection
            Coordinate reference system for plotting (default: Geodetic)
        cross_section_marker_size : int or float
            Size of the station markers on cross-sections (default: 6, smaller than map)
        **kwargs : additional arguments
            Additional arguments passed to ax.scatter()
        """
        # Plot inventory on the main map
        self.map_obj.plot_inventory(inventory, marker_size=marker_size, color=color, 
                                 alpha=alpha, transform=transform, **kwargs)
        
        # Plot inventory on both cross-sections
        self.fig_xs1.plot_inventory(inventory, marker_size=cross_section_marker_size, 
                                   color=color, alpha=alpha, 
                                   **kwargs)
        self.fig_xs2.plot_inventory(inventory, marker_size=cross_section_marker_size, 
                                   color=color, alpha=alpha, 
                                   **kwargs)

    def plot_heatmap(self, *args, **kwargs):
        """
        Plot a heatmap on the map and cross-sections.
        
        This method sends the heatmap data to all three views:
        - Map (fig_m)
        - Cross-section 1 (fig_xs1) 
        - Cross-section 2 (fig_xs2)
        
        Parameters:
        -----------
        *args : 
            Either: catalog (ObsPy Catalog object)
            Or: lat, lon, depth (depth is optional)
        **kwargs : 
            All keyword arguments are passed to the plot_heatmap methods
            of Map and CrossSection classes
        """
        # Plot heatmap on the main map using the stored map object
        self.map_obj.plot_heatmap(*args, **kwargs)
        
        # Plot heatmap on both cross-sections
        self.fig_xs1.plot_heatmap(*args, **kwargs)
        self.fig_xs2.plot_heatmap(*args, **kwargs)

    def title(self, t, x=0.5, y=0.975, fontsize=t1fs, **kwargs):
        self.figure.suptitle(t, x=x, y=y, fontsize=fontsize, **kwargs)

    def subtitle(self, t, x=0.5, y=0.925, fontsize=t2fs, ha='center', va='center', **kwargs):
        self.figure.text(x, y, t, fontsize=fontsize, ha=ha, va=va, **kwargs)

    def text(self, t, x=0.5, y=0.5, fontsize=t2fs, ha='center', va='center', **kwargs):
        self.figure.text(x, y, t, fontsize=fontsize, ha=ha, va=va, **kwargs)

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
        # kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="M{x:1.0f} (357 EQs)",
        #           func=lambda s: np.sqrt(s / .3) / 3
        #           )
        kw = dict(prop="sizes", num=3, fmt="M{x:1.0f}",
                  func=lambda s: np.sqrt(s / .3) / 3
                  # func=lambda s: s * 1
                  )
        legend = self.fig_leg.axes[0].legend(*scatter.legend_elements(**kw),
                                        loc="upper center", bbox_to_anchor=[0.0, 0.0, 1, 1],
                                        title="Magnitude", frameon=False)
        self.fig_leg.add_artist(legend)
        self.fig_leg.axes[0].set_visible(False)
