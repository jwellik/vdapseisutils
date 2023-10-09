"""
Python scripts for plotting earthquake catalogs at volcanoes.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2022 December 19
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patheffects as pe
import matplotlib.projections as proj
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from obspy import UTCDateTime

from vdapseisutils.core.maps import elev_profile
from vdapseisutils.utils.geoutils import sight_point_pyproj, radial_extent2map_extent

# Plotting styles and formatters for maps and cross-sections
plt.rcParams['svg.fonttype'] = 'none'
titlefontsize = t1fs = 16
subtitlefontsize = t2fs = 12
axlabelfontsize = axlf = 8
annotationfontsize = afs = 8
axlabelcolor = axlc = 'grey'

cmap = "viridis_r"
norm = None

default_volcano = {
    'name': "Hood",
    'synonyms': "Wy'east",
    'lat': 45.374,
    'lon': -121.695,
    'elev': 3426,
}


########################################################################################################################
# Development

def scale_bar(ax, length=None, location=(0.15, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.

    Borrowed from: https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length)

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc, fontsize=afs,
            horizontalalignment='center', verticalalignment='bottom')

########################################################################################################################



class ShadedReliefESRI(cimgt.GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url


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
                 # mrange=[-2, 5], msrange=[0, 15],  # results in M0 plotting at ~markersize=6, default (see above)
                 mrange=[-2, 2], msrange=[0, 6],  # defined this way so M-2 is smallest possible event & M2 is ~markersize=6 (default)
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

        if ax==None:
            fig, ax = plt.subplots()

        ax.scatter([0]*len(self.legend_mag), y=self.legend_mag, s=self.legend_s,
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
        ms = m*mag+b  # m*mag+b converts to marker size
        ms[ms < 0] = 0  # marker size must be >=0
        s = ms**2  # convert to point size (**2)
        return s

    def legend_counts(self, cat):
        """COUNTS Counts the number of EQs at each magnitude within the legend scale"""

        nmags = []
        for mag in self.legend_mag:
            rslt = cat[(cat["mag"] >= mag) & (cat["mag"] < mag+1)]
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


def prep_catalog_data_mpl(catalog, s="magnitude", c="time", maglegend=MagLegend(), time_format="matplotlib"):
    """

    PREPCATALOG Converts ObsPy Catalog object to DataFrame w fields appropriate for plotting

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
    catdata["depth"] *= -1  # below sea level values are negative for plotting purposes
    catdata["size"] = MagLegend().mag2s(catdata["mag"])  # converts magnitudes to point size for scatter plot
    return catdata


# UNUSED. Eventually, put this into VolcanoMap
class Map(plt.Axes):
    """ MAP Creates a map for geophysical data """

    name = "map"

    def __init__(self, *args,
                 lat=default_volcano["lat"], lon=default_volcano["lon"], radial_extent_km=50,
                 depth_extent=(-50, 7),
                 map_type='terrain-background', zoom=11,
                 cmap="viridis_r",  # Colorbar for earthquake times
                 maglegend=MagLegend(),
                 title="VOLCANO MAP",
                 subtitle="",
                 figsize=(6, 6), dpi=300,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.maglegend = maglegend

        self.depth_extent = depth_extent  # (bottom_depth, top_altitude)
        self.depth_range = depth_extent[1]-depth_extent[0]  # km

        self.__setup_axis_formatting()
        self.__add_labels_to_xsection()

    # Set up Axis formatting
    def __setup_axis_formatting(self):
        super().tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                            left=False, labelleft=False,
                            bottom=True, labelbottom=True,
                            right=True, labelright=True,
                            top=False, labeltop=False)
        self.axes.yaxis.set_label_position("right")
        super().set_ylabel("Depth", rotation=270, fontsize=axlf, labelpad=10)
        super().set_xlabel("Distance (km)", fontsize=axlf, labelpad=5)


class CrossSection(plt.Axes):
    """
    Creates cross-section for geophysical data

    points: [(centerlat, centerlon), azimuth, radius]
    depth_extent=(-50., 4.),
    n=100,
    download_profile=True,
    label="A",
    maglegend=MagLegend(),
    orientation="horizontal",
    verbose=False,

    * Depths below sea-level are negative
    """

    name = "cross-section"

    # TODO 60km cross-section somewhere in PNW (make this better)
    def __init__(self, *args,
                 points=[(45, -111), -45, 30],  # [(centerlat, centerlon), azimuth, radius]
                 depth_extent=(-50., 4.),
                 n=100,
                 download_profile=True,
                 label="A",
                 maglegend=MagLegend(),
                 orientation="horizontal",
                 verbose=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.verbose = verbose
        self.label = label  # Name of cross-section (eg, A)
        self.full_label = "{a}-{a}'".format(a=self.label)  # eg, A-A'
        self.n = n  # number of points along topographic profile
        self.maglegend = maglegend
        self.orientation = "horizontal"  # horizontal or vertical layout

        self.A1, self.A2, self.az, self.hd = self.__define_line(points)
        self.depth_extent = depth_extent  # (bottom_depth, top_altitude)
        self.depth_range = depth_extent[1]-depth_extent[0]  # km

        self.__setup_axis_formatting()
        self.__add_labels_to_xsection()
        if download_profile:
            self.__download_profile()

    def __define_line(self, points_input):
        """
        Defines a XS profile line given inputs

        Inputs can be:
        - [(lat1, lon1), (lat2, lon2)]
        - [(lat, lon), azimuth, distance] where distance is km
           This extends a line N km at angle azimuth and azimuth+180 from (lat, lon)

        Returns
        """
        # Set A1 & A2, Depth
        if len(points_input) == 2:
            A1 = points_input[0]
            A2 = points_input[1]
            az = 0  # stub  Neds to be azimuth between A1 & A2
            hd = 100  # stub  Needs to be distance between A1 & A2
        elif len(points_input) == 3:
            A1 = sight_point_pyproj(points_input[0], points_input[1], points_input[2]*1000)
            A2 = sight_point_pyproj(points_input[0], points_input[1]+180, points_input[2]*1000)
            az = points_input[1]
            hd = points_input[2]*2  # Horizontal distance
        else:
            print("No points given.")

        return A1, A2, az, hd

    def __setup_axis_formatting(self):
        super().tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                            left=False, labelleft=False,
                            bottom=True, labelbottom=True,
                            right=True, labelright=True,
                            top=False, labeltop=False)
        self.axes.yaxis.set_label_position("right")
        super().set_ylabel("Depth (km)", rotation=270, fontsize=axlf, labelpad=10)
        super().set_xlabel("Distance (km)", fontsize=axlf, labelpad=5)

    # Download topography data along profile
    def __download_profile(self):
        try:
            self.elev_data = elev_profile.download_profile(self.A1, self.A2, n=self.n)  # elevation returned in meters
            plot_profile = True
        # except HTTPError:
        except:
            if self.verbose:
                 print("There was a problem downloading elevation data. Moving on...")
            self.elev_data = []
            plot_profile = False

        if plot_profile:
            # Plot data and format axis for A-A'
            h = self.elev_data['d']
            elev = np.array(self.elev_data['elev']) / 1000  # convert to km
            self.axes.set_xlim([self.elev_data["d"][0], self.elev_data["d"][-1]])
            self.axes.set_ylim(self.depth_extent)
            self.axes.plot_profile(h, elev, color="k", linewidth="0.75")
            # custom spine bounds for a nice clean look
            self.axes.spines['top'].set_visible(False)
            self.axes.spines["left"].set_bounds((self.depth_extent[0], elev[0]))  # depth_extent_v[1] is the top elev
            self.axes.spines["right"].set_bounds(self.depth_extent[0], elev[-1])

    def __add_labels_to_xsection(self):
        # Add labels on cross-section
        x1 = self.hd * 0.05  # a little off from insides
        x2 = self.hd * 0.95  # a little off from insides
        y = self.depth_extent[0] + self.depth_range * 0.05  # a little above bottom

        super().text(x1, y, "{}".format(self.label),
                     fontsize=afs,
                     verticalalignment='bottom', horizontalalignment='left',
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        super().text(x2, y, "{}'".format(self.label),
                     fontsize=afs,
                     verticalalignment='bottom', horizontalalignment='right',
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def get_VE(self):  # return vertical exageration
        #return (plot_height/self.depth_range) / (plot_width/self.hd)  # vertical exaggeration = vertical scale / horizontal scale
        return None

    def project2xs(self, lats, lons):
        """

        PROJECT2XS Project points to cross-section

        returns a column with name "dAA", meaning "distance in km along line AA
        TODO lats and lons must be list like; Make it more flexible
        """

        ## Project EQs to Cross-Sections
        # catdata must be a Pandas DataFrame with "lat" and "lon" columns
        import pyproj
        import math

        # fwdA
        # dA    : EQ distance along A-A'
        # Create projection system
        geodesic = pyproj.Geod(ellps='WGS84')  # Create projection system
        # Angle of cross-section vectors
        fwdAA, backAA, distanceAA = geodesic.inv(self.A1[1], self.A1[0], self.A2[1], self.A2[0])  # (long, lat, long, lat)

        FWDAA = []
        BACKAA = []
        DISTANCEAA = []
        ALPHAAA = []
        DAA = []

        for lat, lon in zip(lats, lons):
            fwdA, backA, distanceA = geodesic.inv(self.A1[1], self.A1[0], lon, lat)  # long, lat, long, lat
            alphaA = fwdA - fwdAA  # angle between A1-pt and A1-A2
            dA = distanceA * math.cos((alphaA) * (np.pi / 180))  # distance to pt along xsection line
            FWDAA.append(fwdA)
            BACKAA.append(backA)
            DISTANCEAA.append(distanceA)
            ALPHAAA.append(alphaA)
            DAA.append(dA)
        # catdata["dAA"] = np.array(DAA) / 1000  # Distance along cross-section A-A' in km
        DAA = np.array(DAA) / 1000

        return DAA

    def project2xs_catdata(self, catdata):
        """

        PROJECT2XS Project points to cross-section

        catdata must be a DataFrame with columns "lat" and "lon"
        returns a column with name "dAA", meaning "distance in km along line AA

        """
        ## Project EQs to Cross-Sections
        # catdata must be a Pandas DataFrame with "lat" and "lon" columns
        import pyproj
        import math

        # fwdA
        # dA    : EQ distance along A-A'
        # Create projection system
        geodesic = pyproj.Geod(ellps='WGS84')  # Create projection system
        # Angle of cross-section vectors
        fwdAA, backAA, distanceAA = geodesic.inv(self.A1[1], self.A1[0], self.A2[1], self.A2[0])  # (long, lat, long, lat)

        FWDAA = []
        BACKAA = []
        DISTANCEAA = []
        ALPHAAA = []
        DAA = []

        for idx, row in catdata.iterrows():
            fwdA, backA, distanceA = geodesic.inv(self.A1[1], self.A1[0], row["lon"], row["lat"])  # long, lat, long, lat
            alphaA = fwdA - fwdAA  # angle between A1-pt and A1-A2
            dA = distanceA * math.cos((alphaA) * (np.pi / 180))  # distance to pt along xsection line
            FWDAA.append(fwdA)
            BACKAA.append(backA)
            DISTANCEAA.append(distanceA)
            ALPHAAA.append(alphaA)
            DAA.append(dA)
        catdata["dAA"] = np.array(DAA) / 1000  # Distance along cross-section A-A' in km

        return catdata

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

        x = self.project2xs(catdata["lat"], catdata["lon"])
        super().scatter(x, catdata["depth"], s=s, c=c, cmap=cmap, alpha=alpha, **kwargs)

    def plot_inventory(self, inventory, **kwargs):
        print("NOTE: PLOT_INVENTORY not yet implemented.")
        # Convert inventory to df
        # Project inventory lat,lon to profile
        # z = np.array(z) / 1000 * -1  # Convert to km and make negative for plotting purposes
        # super().plot(x, z, **kwargs)

    def plot_profile(self, x, elev, *args, **kwargs):
        """ PLOT_PROFILE Plots elevation profile across cross-section"""
        super().plot(x, elev, *args, **kwargs)  # Plot pr

    def plot(self, lat, lon, z, z_dir="depth", z_unit="m", *args, **kwargs):
        # Assumes z is given as meters depth (down)
        # Creates a negative axis
        # E.g., 5000 m depth is plotted as -5
        # E.g., 3200 m elevation is plotted as 3.2

        z_unit_conv = 1000 if z_unit == "m" else 1
        z_dir_conv  = -1 if z_dir == "depth" else 1

        lat = lat  # Should this be put into a list or not?
        lon = lon
        z = np.array(z)/z_unit_conv*z_dir_conv  # Convert to km and make negative for plotting purposes
        x = self.project2xs(lat, lon)  # returned as array
        super().plot(x, z, *args, **kwargs)

    def scatter(self, lat, lon, z, z_dir="depth", z_unit="m", **kwargs):
        # Assumes z is given as meters depth (down)
        # Creates a negative axis
        # E.g., 5000 m depth is plotted as -5
        # E.g., 3200 m elevation is plotted as 3.2

        # TODO Why is scatter not converting meters to km?
        z_unit_conv = 1000 if z_unit == "m" else 1
        z_dir_conv  = -1 if z_dir == "depth" else 1

        daa = self.project2xs(lat, lon)  # Put lat, lon into a list or not?
        super().scatter(daa, z*z_dir_conv, **kwargs)

    def info(self):
        print("::: Cross-Section ({})".format(self.label))
        print("    - P1-P2 : (lat,lon)-(lat,lon) (... km @ ... deg) ")
        print("    - depth_extent : (lat,lon)")
        print("    - # EQs : ")
        print()


class TimeSeries(plt.Axes):
    """
    Creates time-series plot for geophysical data

    * Depths below sea-level are negative
    """

    name = "time-series"

    def __init__(self, *args,
                 trange = None,
                 depth_extent=(-50., 4.),
                 maglegend=MagLegend(),
                 colorbar=False,
                 verbose=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.trange = trange
        self.depth_extent = depth_extent
        self.maglegend = maglegend
        self.orientation = "horizontal"  # horizontal or vertical layout

        # if colorbar:
        #     axC = fig.add_axes(cbar_pos)  # Colorbar

        # TimeSeries Axis
        if colorbar:
            self.tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                            left=True, labelleft=True,
                            bottom=False, labelbottom=False,
                            right=False, labelright=False,
                            top=False, labeltop=False)
            self.axC.tick_params(labelsize=axlf, labelcolor=axlc)  # How to get to colorbar axis from within TS axes?
            self.axC.set_visible(True)
        else:
            self.tick_params(axis='both', labelsize=axlf, labelcolor=axlc,
                            left=True, labelleft=True,
                            bottom=True, labelbottom=True,
                            right=False, labelright=False,
                            top=False, labeltop=False)
        self.set_ylabel("Depth (km)", fontsize=axlf, labelpad=0)
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
            self.xaxis.set_major_locator(loc)
            self.xaxis.set_major_formatter(formatter)
        # self.set_xlim([tmin, tmax])  # Set time extent of time series axis (as mpldate)


    def scatter(self, t, y, yaxis="Depth", **kwargs):
        super().scatter(t, y, **kwargs)

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

        super().scatter(catdata["time"], catdata["depth"], s=s, c=c, **kwargs)

        # Calculate time range
        tmin = UTCDateTime(self.trange[0]) if self.trange is not None else catdata.iloc[0]["time"]
        tmax = UTCDateTime(self.trange[-1]) if self.trange is not None else catdata.iloc[-1]["time"]
        # super().set_xlim([tmin, tmax])


class VolcanoMap(Figure):

    proj.register_projection(CrossSection)
    proj.register_projection(TimeSeries)

    def __init__(self, *args,
                 lat=default_volcano["lat"], lon=default_volcano["lon"], radial_extent_km=50,
                 xs1=(225, 30, "A"),  # Define Cross Section 1: start angle, radius, name; e.g., 225d from North at radius of 30 km (60 km long) (SW-NE)
                 xs2=(315, 30, "B"),  # Define Cross Section 2: start angle, radius, name; e.g., 315d from North at radius of 30 km (60 km long) (NW-SE)
                 map_type='terrain-background', zoom=11,
                 depth_extent=(-50, 7),
                 cmap="viridis_r",  # Colorbar for earthquake times
                 trange=None,  # (start, stop) for time series colorbar
                 maglegend=MagLegend(),
                 title="VOLCANO MAP",
                 subtitle="",
                 figsize=(6, 6), dpi=300,
                 **kwargs):
        super().__init__(*args, **kwargs)

        ## Calculate map properties
        self.origin = (lat, lon)
        self.map_extent = radial_extent2map_extent(lat, lon, radial_extent_km)  # returns LRBT
        # tmin = UTCDateTime(trange[0]) if trange is not None else catalog[0].origins[-1].time
        # tmax = UTCDateTime(trange[-1]) if trange is not None else catalog[0].origins[-1].time
        self.radial_extent_km = radial_extent_km
        self.depth_extent = depth_extent
        self.title = title
        self.subtitle = subtitle
        self.zoom = zoom
        self.figsize = figsize
        self.dpi = dpi
        self.kwargs = kwargs
        # self.stats = dict({"nEarthquakes": np.nan()})  # Dictionary with statistics about map data

        ## Download basemap tiles
        try:
            import urllib2
        except:
            pass

        self.tiles = cimgt.Stamen("terrain-background", desired_tile_form="L")
        self.__pos = self.__define_axis_positions()
        self.__make_all_axes(lat, lon, radial_extent_km, depth_extent, xs1, xs2)  # makes self.axm, self.xs1, self.xs2, self.axT, self.axL, self.axC
        self.__set_title()
        self.__set_gridlines_and_labels()  # Set Axis Tick Positions & Gridlines
        self.__set_axes_extents()  # Set extent of axes in physical units
        # try:
        #     self.axm.add_image(self.tiles, zoom, cmap="Greys_r")  # Add basemap
        # except:
        #     print("Error: Problem plotting basemap w Cartopy.")
        self.__add_xsections_to_map()  # Add both cross-sections to mapview

        scale_bar(self.axes[0], 10)


    def info(self):

        print('::: {} (MapFigure) :::'.format(self.title))
        print('      {}'.format(self.subtitle))
        print('      origin        : {}'.format(self.origin))
        print('      radial_extent : {} km'.format(self.radial_extent_km))
        print('      depth_extent  : {} km'.format(self.depth_extent))
        # print('      EQs on map    : {} '.format(len(catalog)))
        # print("::: Magnitude Distribution")  # Print Catalog information
        # for M, N in zip(scale_mag, nmags):
        #     print("       - M{:2d}  : {: 5d}".format(M, N))
        # if nremoved > 0:
        #     print('       ({} not within map/time bounds, removed)'.format(nremoved))
        print()


    def plot(self, lat, lon, depth, *args, **kwargs):
        # Plot to map
        self.figure.axes[0].plot(lon, lat, *args, transform=ccrs.Geodetic(), **kwargs)  # GeoAxes.plot() takes args as x,y

        # Plot to cross section
        self.xs1.plot(lat, lon, depth, *args, **kwargs)  # CrossSection axes takes args as lat, lon, depth (positive down)
        self.xs2.plot(lat, lon, depth, *args, **kwargs)  # CrossSection axes takes args as lat, lon, depth (positive down)


    def scatter(self, lat, lon, depth, **kwargs):
        self.figure.axes[0].scatter(lon, lat, transform=ccrs.Geodetic(), **kwargs)
        self.xs1.scatter(lat, lon, depth, **kwargs)
        self.xs2.scatter(lat, lon, depth, **kwargs)


    def plot_catalog(self, catalog, s="magnitude", c="time", cmap="viridis_r", alpha=0.5, **kwargs):

        # from obspy import UTCDateTime
        # import matplotlib as mpl
        #
        # tmin = UTCDateTime(trange[0]) if trange is not None else catalog[0].origins[-1].time
        # tmax = UTCDateTime(trange[-1]) if trange is not None else catalog[0].origins[-1].time
        # norm = norm = mpl.colors.Normalize(vmin=tmin.matplotlib_date, vmax=tmax.matplotlib_date)

        # Retrieves catalog data for plotting with proper size and colors pre-assigned
        catdata = prep_catalog_data_mpl(catalog, MagLegend)
        s_input = s
        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        c_input = c
        if c == "time":
            c = catdata["time"]
        else:
            c = c

        # x = self.project2xs(catdata)
        self.figure.axes[0].scatter(catdata["lon"], catdata["lat"], transform=ccrs.Geodetic(), s=s, c=c, cmap=cmap, alpha=alpha, **kwargs)
        self.xs1.plot_catalog(catalog, s=s_input, c=c_input, cmap=cmap, alpha=alpha, **kwargs)
        self.xs2.plot_catalog(catalog, s=s_input, c=c_input, cmap=cmap, alpha=alpha, **kwargs)
        self.axT.plot_catalog(catalog, s=s_input, c=c_input, cmap=cmap, alpha=alpha, **kwargs)


    def plot_inventory(self, inventory, linewidth=0, marker="v", color="k", markersize=4, **kwargs):
        print("WARNING - PLOT_INVENTORY noy yet complete.")
        from vdapseisutils.utils.obspyutils.inventoryutils import inventory2df
        df = inventory2df(inventory)
        self.figure.axes[0].plot(df["longitude"], df["latitude"], transform=ccrs.Geodetic(), linewidth=linewidth, marker=marker, color=color, markersize=markersize, **kwargs)


    def add_legend(self, lines=None, labels=None, bbox_to_anchor=[1.05, 1.05, .25, .25]):
        # handles, labels = ax.get_legend_handles_labels()
        # or
        # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        if lines is None:
            lines_labels = [ax.get_legend_handles_labels() for ax in [self.axm]]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # self.figure.legend(lines, labels, bbox_to_anchor=[1.5,1.1], bbox_transform=self.figure.transFigure, frameon=False)
        # self.figure.legend(lines, labels, loc="lower right")
        self.figure.legend(lines, labels, bbox_transform=self.axL.transAxes, bbox_to_anchor=[1.25,0,0.5,1.0], frameon=False)

    def __add_xsection_from_CrossSection(self):
        pass

    def __add_xsection_from_points(self):
        pass

    def __add_xsections_to_map(self):

        # Plot cross-section lines on map
        self.figure.axes[0].plot([self.xs1.A1[1], self.xs1.A2[1]], [self.xs1.A1[0], self.xs1.A2[0]], linewidth=0.75, color="k", transform=ccrs.Geodetic())
        self.figure.axes[0].plot([self.xs2.A1[1], self.xs2.A2[1]], [self.xs2.A1[0], self.xs2.A2[0]], linewidth=0.75, color="k", transform=ccrs.Geodetic())

        # Plot cross-section labels on map
        self.axm.text(self.xs1.A1[1], self.xs1.A1[0], "{}".format(self.xs1.label), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        self.axm.text(self.xs1.A2[1], self.xs1.A2[0], "{}'".format(self.xs1.label), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        self.axm.text(self.xs2.A1[1], self.xs2.A1[0], "{}".format(self.xs2.label), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        self.axm.text(self.xs2.A2[1], self.xs2.A2[0], "{}'".format(self.xs2.label), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])


    def __define_axis_positions(self):

        ## Define Axis Positions
        # define axes positions (inches) LBWH
        s = 12  # square dimension (h == w)
        cw = 11  # content_width - smaller than figure size for padding
        ch = 9  # content_height
        fig_pos = dict()
        fig_pos["map"]  = list(np.array([1, 4+((s-9)/2), 5, 5])/s)
        fig_pos["xs1"]  = list(np.array([6.25, 7+((s-9)/2), 4.5, 2])/s)
        fig_pos["xs2"]  = list(np.array([6.25, 4+((s-9)/2), 4.5, 2])/s)
        fig_pos["ts"]   = list(np.array([1, 0.3+((s-9)/2), 7.5, 2.7])/s)
        fig_pos["cbar"] = list(np.array([1, 0+((s-9)/2), 7.5, 0.3])/s)
        fig_pos["legend"] = list(np.array([9, 0+((s-9)/2), 1.25, 3])/s)
        fig_pos["title"] = list(np.array([6, 10+((s-9)/2), 12, 1])/s)
        fig_pos["subtitle"] = list(np.array([6, 9.5+((s-9)/2), 12, 0.5])/s)

        return fig_pos


    def __make_all_axes(self, lat, lon, radial_extent_km, depth_extent, xs1, xs2):
        # Make axes
        self.axm = self.add_axes(self.__pos["map"], projection=self.tiles.crs)
        self.xs1 = self.add_axes(self.__pos["xs1"], projection='cross-section',
                           points=[(lat, lon), xs1[0], xs1[1]],
                           depth_extent=depth_extent,
                           label=xs1[2],
                           )
        self.xs2 = self.add_axes(self.__pos["xs2"], projection='cross-section',
                           points=[(lat, lon), xs2[0], xs2[1]],
                           depth_extent=depth_extent,
                           label=xs2[2],
                           )
        self.axT = self.add_axes(self.__pos["ts"], projection="time-series")   # Time Series (Depth v Time)
        self.axL = self.add_axes(self.__pos["legend"])  # Legend (Magnitude scale)
        # self.axC = self.add_axes(self.__pos["cbar"])  # Colorbar
        self.axL.set_visible(False)  # Should be false eventually
        # self.axC.set_visible(False)


    def __set_gridlines_and_labels(self):
        # Map gridlines
        glv = self.axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
        glv.top_labels = False
        glv.bottom_labels = True
        glv.left_labels = True
        glv.right_labels = False
        glv.xlines = True
        glv.xlabel_style = {'size': axlf, 'color': axlc}
        glv.ylabel_style = {'size': axlf, 'color': axlc}


    def __set_axes_extents(self):
        ### __SET_AXES_EXTENTS Set XSection, Time Series Depth Extent
        self.figure.axes[0].set_extent(self.map_extent, crs=ccrs.Geodetic())
        self.figure.axes[1].set_ylim([self.xs1.depth_extent[0], self.xs1.depth_extent[1]])
        self.figure.axes[2].set_ylim([self.xs2.depth_extent[0], self.xs2.depth_extent[1]])
        self.axT.set_ylim([self.depth_extent[0], self.depth_extent[1]])  # Why isn't this self.axT.... ?


    def __set_title(self):
        # Title as custom text
        self.text(self.__pos["title"][0], self.__pos["title"][1], self.title, fontsize=t1fs,
                 verticalalignment='center', horizontalalignment='center')
        self.text(self.__pos["subtitle"][0], self.__pos["subtitle"][1], self.subtitle, fontsize=t2fs,
                 verticalalignment='center', horizontalalignment='center')
