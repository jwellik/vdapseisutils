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
titlefontsize = t1fs = 16
subtitlefontsize = t2fs = 12
axlabelfontsize = axlf = 10
annotationfontsize = afs = 10
axlabelcolor = axlc = 'grey'

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
                        resolution="01d",
                        topo=True, bath=False, radiance=[315, 60],
                        vertical_exag=1.5,
                        cmap="Greys_r", alpha=0.50):
    """
    ADD_HILLSHADE_PYGMT Adds a hillshade to a Cartopy GeoAxes using data from PyGMT's load_earth_relief()

    The final image is an overlay of the hillshade (75%) and the original DEM (25%)

    Downloads srtm data using PyGMT's load_earth_relief function.
    Use PyGMT to create the hillshade.
    Then, place elevation data into to NumPy grid with shape (n,m,3)
        n : number of cells in the latitude direction
        m : number of cells in the longitude direction
        3 : Each n,m gridcell contains (lat, lon, elevation)
    Mask hillshade values for elevations less than zero.
    Plot NumPy hillshade grid using imshow()
        - I'm not sure why the grid needs to be rotated; plotting the grid w contour() does not require this
        - .T[2] isolates the elevation layer

    PyGMT's load_earth_relief(): https://www.pygmt.org/dev/api/generated/pygmt.datasets.load_earth_relief.html
    """

    import numpy as np
    import pygmt

    # Download PyGMT earth relief data, create hillshade
    srtm = pygmt.datasets.load_earth_relief(region=extent, data_source=data_source, resolution=resolution)
    srtm.data = srtm.data * vertical_exag  # Multiply by vertical exaggeration
    srtm_hs = pygmt.grdgradient(grid=srtm, radiance=radiance)   # Compute hillshade on the exaggerated DEM
    # srtm_hs = pygmt.grdgradient(grid=srtm, azimuth="0/90", normalize="t1")
    # srtm_hs = pygmt.grdgradient(grid=srtm, radiance=[315, 45])

    # create arrays with all lon/lat values from min to max and
    # origin of data grid as stated in SRTM data file header
    lats = srtm.lat.data  # numpy.ndarray (n,)  should be [maxlat:minlat]
    lons = srtm.lon.data  # numpy.ndarray (m,)

    # create grids and compute map projection coordinates for lon/lat grid
    # use grid_elev to remove hillshade value from cells below sea level
    projection = ccrs.PlateCarree()
    grid_elev = projection.transform_points(ccrs.Geodetic(),
                                            *np.meshgrid(lons, lats),  # unpacked x, y
                                            srtm.data)  # z from topo data  # should be <class 'numpy.ndarray'> (n,m)
    grid_hs = projection.transform_points(ccrs.Geodetic(),
                                          *np.meshgrid(lons, lats),  # unpacked x, y
                                          srtm_hs.data)  # z from topo data  # should be <class 'numpy.ndarray'> (n,m)

    # Hillshades scale from -1 to 1; 1 is the brightest color (white), so use that for below sea level
    if not bath:
        grid_hs[grid_elev <= 0] = 1  # Remove hillshade values for cells w elevation below sea level
        grid_elev[grid_elev <= 0] = 0
    if not topo:
        grid_hs[grid_elev >= 0] = 1  # Remove hillshade values for cells w elevation above sea level
        grid_elev[grid_elev >= 0] = 0

    # Add hillshade
    # - just the hillshade
    grid_final = grid_hs.T[2]
    # - alternatively, hillshade + DEM
    # matrix_min = np.min(grid_elev.T[2])
    # matrix_max = np.max(grid_elev.T[2])
    # grid_elev_norm = 2 * (grid_elev.T[2] - matrix_min) / (matrix_max - matrix_min) - 1  # normalize elevation data between -1 and 1
    # grid_final = grid_hs.T[2] * 0.75 + grid_elev_norm * 0.25  # hill-shade + DEM
    ax.imshow(np.rot90(grid_final), extent=extent, transform=projection, cmap=cmap, alpha=alpha)

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
                 dpi=300, **kwargs):

        # Initialize the Figure object
        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure(figsize=(8, 8), dpi=dpi)
        super().__init__(*args, **kwargs)
        self.__dict__ = fig.__dict__
        self._dpi = dpi  # plt.show() produces an error without this. Problematic???

        # Create the principal axes for the map
        # self.ax = self.add_subplot(*args, projection=ShadedReliefESRI().crs, **kwargs)  # creates principal GeoAxes
        self.ax = self.add_subplot(*args, projection=ccrs.Mercator(), **kwargs)  # creates principal GeoAxes

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
        glv.xlabel_style = {'size': axlf, 'color': axlc}
        glv.ylabel_style = {'size': axlf, 'color': axlc}

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

    def add_hillshade(self, source="PyGMT", data_source="igpp", resolution="01s", topo=True, bath=False, radiance=[315, 60], alpha=0.50):
        if source.lower() == "pygmt":
            try:
                self.ax = add_hillshade_pygmt(self.ax, extent=self.properties["map_extent"], data_source=data_source,
                                              resolution=resolution, topo=topo, bath=bath,
                                              radiance=radiance, alpha=alpha)
            except:
                print("Failed to add hillshade via PyGMT. Moving on.")
        else:
            print("Elevation & hillshade source '{}' not available.".format(source))

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
                  label=None, fontsize=afs, va='center', ha='center',
                  transform=ccrs.Geodetic(), **kwargs):
        label0 = label if label else ""  # Example: A
        label1 = label+"'" if label else ""  # Example: A'
        self.ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=linewidth,
                     transform=transform, **kwargs)
        self.ax.text(p1[1], p1[0], label0, fontsize=fontsize, ha=ha, va=va, color=color,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                     transform=transform)
        self.ax.text(p2[1], p2[0], label1, fontsize=fontsize, ha=ha, va=va, color=color,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                     transform=transform)

    # def plot_inventory(self, inventory, "vk", transform=ccrs.Geodetic(), **kwargs):
    #     self.ax.plot(lon, lat, transform=transform)


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
                 maglegend=MagLegend(),
                 verbose=False,
                 figsize=(10, 6), dpi=300,
                 **kwargs):

        # Initialize the figure
        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure(figsize=(8, 2), dpi=300)
        super().__init__(*args, **kwargs)
        self.__dict__ = fig.__dict__
        self._dpi = 300  # plt.show() produces an error without this. Problematic???
        self.ax = self.add_subplot()  # Creates the principal axes

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

        self.properties["orientation"] = "horizontal"  # horizontal or vertical layout

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

        # Plot data and format axis for A-A'
        hd = self.profile.distance / 1000  # horizontal distance along line (convert meters to km)
        elev = np.array(self.profile.elevation / 1000)  # elevation (convert m to km)
        self.ax.set_xlim([hd[0], hd[-1]])
        self.ax.set_ylim(self.properties["depth_extent"])
        self.plot(x=hd, z=elev, z_dir="elev", color="k", z_unit="km",
                  linewidth=0.75)  # Use CrossSection.plot() instead of Axes.plot()

        # custom spine bounds for a nice clean look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines["left"].set_bounds(
            (self.properties["depth_extent"][0], elev[0]))  # depth_extent_v[1] is the top elev
        self.ax.spines["right"].set_bounds(self.properties["depth_extent"][0], elev[-1])

    def __setup_axis_formatting(self):
        self.ax.tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                            left=False, labelleft=False,
                            bottom=True, labelbottom=True,
                            right=True, labelright=True,
                            top=False, labeltop=False)
        self.ax.yaxis.set_label_position("right")
        self.ax.set_ylabel("Depth (km)", rotation=270, fontsize=axlf, labelpad=15)
        self.ax.set_xlabel("Distance (km)", fontsize=axlf, labelpad=10)

    def __add_labels_to_xsection(self):
        # Add labels on cross-section
        x1 = self.profile.length / 1000 * 0.05  # a little off from insides (convert m to km)
        x2 = self.profile.length / 1000 * 0.95  # a little off from insides (convert m to km)
        y = self.properties["depth_extent"][0] + self.properties["depth_range"] * 0.05  # a little above bottom

        self.ax.text(x1, y, "{}".format(self.properties["label"]),
                     fontsize=afs,
                     verticalalignment='bottom', horizontalalignment='left',
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        self.ax.text(x2, y, "{}'".format(self.properties["label"]),
                     fontsize=afs,
                     verticalalignment='bottom', horizontalalignment='right',
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def get_ve(self):  # return vertical exageration
        # return (plot_height/self.depth_range) / (plot_width/self.hd)  # vertical exaggeration = vertical scale / horizontal scale
        print("GET_VE not yet implemented :-(")

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

    def plot_inventory(self, inventory, **kwargs):
        print("NOTE: PLOT_INVENTORY not yet implemented.")
        # Convert inventory to df
        # Project inventory lat,lon to profile
        # z = np.array(z) / 1000 * -1  # Convert to km and make negative for swarmmpl purposes
        # super().plot(x, z, **kwargs)

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
            x = project2line(lat, lon, P1=self.A1, P2=self.A2)  # Put lat, lon into a list or not?

        self.ax.scatter(x, z, *args, **kwargs)

    def info(self):
        print("::: Cross-Section ({}) :::".format(self.properties["label"]))
        # print("    - P1-P2 : (lat,lon)-(lat,lon) (... km @ ... deg) ")
        # print("    - depth_extent : (lat,lon)")
        # print("    - # EQs : ")
        print()


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
                 **kwargs):

        # Initialize Figure
        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure(figsize=(8, 2), dpi=300)
        super().__init__(*args, **kwargs)
        self.__dict__ = fig.__dict__
        self._dpi = 300  # plt.show() produces an error without this. Problematic???
        self.ax = self.add_subplot()  # Creates the principal axes

        self.trange = trange
        self.depth_extent = depth_extent
        self.maglegend = maglegend
        self.orientation = "horizontal"  # horizontal or vertical layout
        self.axis_type = axis_type
        self.scatter_unit = "magnitude"  # not yet implemented

        # if colorbar:
        #     axC = fig.add_axes(cbar_pos)  # Colorbar

        # TimeSeries Axis
        if colorbar:
            self.ax.tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                                left=True, labelleft=True,
                                bottom=False, labelbottom=False,
                                right=False, labelright=False,
                                top=False, labeltop=False)
            self.axC.tick_params(labelsize=axlf, labelcolor=axlc)  # How to get to colorbar axis from within TS axes?
            self.axC.set_visible(True)
        else:
            self.ax.tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                                left=True, labelleft=True,
                                bottom=True, labelbottom=True,
                                right=False, labelright=False,
                                top=False, labeltop=False)

        # Set Y-Label
        if self.axis_type == "depth":
            ylabel = "Depth (km)"
        elif self.axis_type == "magnitude":
            ylabel = "Magnitude"
        self.ax.set_ylabel(ylabel, fontsize=axlf, labelpad=15)
        # self.set_ylim([y_extent[0], y_extent[1]])

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

    def scatter(self, t, y, yaxis="Depth", **kwargs):
        self.ax.scatter(convert_timeformat(t, "matplotlib"), y, **kwargs)

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
                 **kwargs):

        # Initialize the Figure object
        figsize = 6
        figscale = 8
        self.figscale = figscale
        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure(figsize=(figsize, figsize), dpi=300)
        super().__init__(*args, **kwargs)
        self.__dict__ = fig.__dict__
        self._dpi = 300  # plt.show() produces an error without this. Problematic???

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
        spec = fig.add_gridspec(1, 1)

        # subfigure - Map
        self.fig_m = fig.add_subfigure(spec[0:1, 0:1])
        self.fig_m = Map(fig=self.fig_m, origin=origin, radial_extent_km=radial_extent_km)
        self.fig_m.add_hillshade()
        self.fig_m.add_scalebar()
        lbwh = np.array([0.9, 4.0, 3.0, 3.0])
        self.fig_m.ax.set_position(lbwh / figscale)

        # subfigure - Top Cross-Section
        self.fig_xs1 = fig.add_subfigure(spec[0:1, 0:1])
        # self.fig_xs1 = CrossSection(fig=self.fig_xs1, origin=origin, radius_km=radial_extent_km,
        #                        depth_extent=depth_extent, azimuth=270, label="A")
        self.fig_xs1 = CrossSection(fig=self.fig_xs1, **{**xs1_defaults, **xs1})  # overwrite defaults w user input
        lbwh = np.array([4.0, 5.8, 3.0, 1.2])
        self.fig_xs1.ax.set_position(lbwh / figscale)

        # subfigure - Bottom Cross-Section
        self.fig_xs2 = fig.add_subfigure(spec[0:1, 0:1])
        # self.fig_xs2 = CrossSection(fig=self.fig_xs2,  origin=origin, radius_km=radial_extent_km,
        #                        depth_extent=depth_extent, azimuth=0, label="B")
        self.fig_xs2 = CrossSection(fig=self.fig_xs2, ** {**xs2_defaults, **xs2})  # overwrite defaults w user input
        lbwh = np.array([4.0, 4.0, 3.0, 1.2])
        self.fig_xs2.ax.set_position(lbwh / figscale)

        # subfigure - TimeSeries
        self.fig_ts = fig.add_subfigure(spec[0:1, 0:1])
        self.fig_ts = TimeSeries(fig=self.fig_ts, depth_extent=(-5, 4), axis_type=self.properties["ts_axis_type"])
        lbwh = np.array([0.9, 1.2, 4.5, 2.1])
        self.fig_ts.ax.set_position(lbwh / figscale)

        # subfigure - Legend
        self.fig_leg = fig.add_subfigure(spec[0:1, 0:1])
        axL = self.fig_leg.add_subplot(111)
        lbwh = np.array([5.5, 1.2, 1.9, 2.1])
        axL.set_position(lbwh / figscale)
        # axL.set_visible(False)

        # Plot Cross-Section lines to Map
        self.fig_m.plot_line(self.fig_xs1.properties["points"][0], self.fig_xs1.properties["points"][1], label=self.fig_xs1.properties["label"])
        self.fig_m.plot_line(self.fig_xs2.properties["points"][0], self.fig_xs2.properties["points"][1], label=self.fig_xs2.properties["label"])


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
        self.fig_m.plot_catalog(*args, transform=transform, **kwargs)
        self.fig_xs1.plot_catalog(*args, **kwargs)
        self.fig_xs2.plot_catalog(*args, **kwargs)
        self.fig_ts.plot_catalog(*args, **kwargs)

    def title(self, t, x=0.5, y=0.975, fontsize=t1fs, **kwargs):
        self.figure.suptitle(t, x=x, y=y, fontsize=fontsize, **kwargs)

    def subtitle(self, t, x=0.5, y=0.925, fontsize=t2fs, ha='center', va='center', **kwargs):
        self.figure.text(x, y, t, fontsize=fontsize, ha=ha, va=va, **kwargs)

    def text(self, t, x=0.5, y=0.5, fontsize=t2fs, ha='center', va='center', **kwargs):
        self.figure.text(x, y, t, fontsize=fontsize, ha=ha, va=va, **kwargs)

    def reftext(self, t, x=0.025, y=0.025, fontsize=afs, color="grey", ha="left", va="center", **kwargs):
        super().text(x, y, t, fontsize=fontsize, color=color, ha=ha, va=va, **kwargs)

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
                                        title="Magnitude", fontsize=afs, frameon=False)
        self.fig_leg.add_artist(legend)
        self.fig_leg.axes[0].set_visible(False)
