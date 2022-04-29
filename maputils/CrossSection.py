import matplotlib.pyplot as plt
import numpy as np
from vdapseisutils.maputils.utils import elev_profile


class CrossSection:

    # XSection Properties
    points = []
    name = "A-A'"

    # Plot properties
    figsize = (8, 4)
    orientation = "horizontal"
    title = "Cross Section A-A'"
    lunits = "meters"
    linewidth = 0.75
    color = 'k'

    def __init__(self, A1, A2,
                 n=100,
                 depth_extent=(-50., 4.),  # -> float # km (bottom_depth, top_altitude)
                 ):

        self.points = [A1, A2]
        self.A1 = dict({"lat": A1[0], "lon": A1[1]})  # for convenience
        self.A2 = dict({"lat": A2[0], "lon": A2[1]})  # for convenience
        self.depth_extent = depth_extent  # (bottom_depth, top_altitude)

        # Download topography data along profile
        self.lat, self.lon, self.d, self.elev = elev_profile.download_profile(self.points[0], self.points[1], n=n)  # elevation returned in meters
        self.elev = np.array(self.elev) / 1000  # convert m to km

    def plot_original(self):

        # Prepare the figure
        self.fig = plt.figure(figsize=self.figsize)

        # Define the length units
        # TODO Create self.length = [], self.lengthlats = [], self.lengthlons = []
        #   Always populate all three; plotting uses length for the axis units, but uses desired units for plot labels
        #   self.length used to determine cut-off for proj/rotation of pts to line
        if self.length == 'lon' or self.length == 'longitude':
            dist = self.lon
        elif self.length == 'lat' or self.length == 'latitude':
            dist = self.lat
        elif self.length == 'distance' or self.length == 'kilometers':
            dist = self.d

        # Plot the data and configure axes
        if self.orientation == 'horizontal':
            plt.plot(dist, self.elev, color=self.color, linewidth=self.linewidth)
            self.fig.axes[0].spines['top'].set_visible(False)  # custom spine bounds for a nice clean look
            self.fig.axes[0].spines.left.set_bounds((self.depth_extent[0], self.elev[0]))  # depth_extent_v[1] is the top elev
            self.fig.axes[0].spines.right.set_bounds((self.depth_extent[0], self.elev[-1]))
            self.fig.axes[0].set_xlim((dist[0], dist[-1]))
            self.fig.axes[0].set_ylim(self.depth_extent)
        elif self.orientation == 'vertical':
            plt.plot(self.elev, dist, color=self.color, linewidth=self.linewidth)
            self.fig.axes[0].spines['left'].set_visible(False)  # custom spine bounds for a nice clean look
            self.fig.axes[0].spines.bottom.set_bounds((self.depth_extent[0], self.elev[0]))  # depth_extent_v[1] is the top elev
            self.fig.axes[0].spines.top.set_bounds((self.depth_extent[0], self.elev[-1]))
            self.fig.axes[0].set_ylim((dist[0], dist[-1]))
            self.fig.axes[0].set_xlim([self.depth_extent[1], self.depth_extent[0]])
            self.fig.axes[0].yaxis.tick_right()

        plt.draw()

    def plot(self, lat, lon, depth, **kwargs):
        pass

    def scatter(self, lat, lon, depth, **kwargs):
        pass

    def proj2line(self, lat, lon):
        """
        - Creates a projection system based on the first point of the line (A)
        - Transforms all points to that projection system
        - Rotate points to line
        - Plots points accordingly

        EPSG:4326 - WGS84, latitude/longitude coordinate system based on the Earth's center of mass,
            used by the Global Positioning System among others.

        Resources:
        https://geopandas.org/en/stable/docs/user_guide/projections.html
        https://gis.stackexchange.com/questions/330445/generating-custom-projection-in-pyproj
        My old Matlab code for "wingplot"

        Unused but potentially useful resources:
        https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        https://scikit-spatial.readthedocs.io/en/stable/gallery/projection/plot_point_line.html

        :param lat:
        :param lon:
        :return:
        """

        import pyproj
        import math

        # Prepare data
        # - ensure that lat, lon are lists

        # Create projection system
        wgs = pyproj.Proj('epsg:4326')  # assuming you're using WGS84 geographic
        epsg = pyproj.CRS("+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs").format(
            lat=self.A1["lat"], lon=self.A1["lon"]
        ).to_epsg()
        crs = pyproj.Proj(epsg)

        # Transform points to projection system
        self.A1["x"], self.A1["y"] = pyproj.transform(wgs, crs, self.A1["lat"], self.A1["lon"])
        self.A2["x"], self.A2["y"] = pyproj.transform(wgs, crs, self.A2["lat"], self.A2["lon"])
        x, y = pyproj.transform(wgs, crs, lat, lon)
        # Note: The distance between self.A1["x", "y"] and self.A2["x", "y"] should be == self.length

        # Rotate points to line
        # - Normalize to A1 (A1 = (0,0))
        # - Rotate
        # - Remove points that are not within bounds (?)

        # % earthquakes (or any point)
        # x0 = x - vx;
        # y0 = y - vy; % adjust eq coordinates relative to volcano
        # a = sqrt(x0. ^ 2 + y0. ^ 2); % distance from origin (volcano) to point(earthquake)
        # angle = atan2(y0, x0); % angle from volcano (origin) to eq
        # phiAA = angle - params.angleA; % angle between xsection vector and vector to point
        # phiBB = angle - params.angleB;
        # AA0 = a. * cos(phiAA); % length along xsection A - A' from volcano (origin)
        # BB0 = -1 * (a. * cos(phiBB)); % length along cross section B - B' from volcano (origin)
        x0 = x - self.A1["x"]  # Adjust eq coordinates to A1
        y0 = y - self.A1["y"]
        a = math.sqrt(x0**2 + y0**2)  # distance from A1 to eq point
        angle = math.atan2(y0, x0)  # angle from A1 to eq
        phiAA = angle - self.theta  # angle between xsection vector and vector to point
        AA0 = a * math.cos(phiAA)  # length along xsection A-A' from A1

        return AA0

    def scikit_proj2line(self, lat, lon):
        """
        - Creates a projection system based on the first point of the line (A)
        - Transforms all points to that projection system
        - Rotate points to line
        - Plots points accordingly

        EPSG:4326 - WGS84, latitude/longitude coordinate system based on the Earth's center of mass,
            used by the Global Positioning System among others.

        Resources:
        https://geopandas.org/en/stable/docs/user_guide/projections.html
        https://gis.stackexchange.com/questions/330445/generating-custom-projection-in-pyproj

        Unused but potentially useful resources:
        https://scikit-spatial.readthedocs.io/en/stable/gallery/projection/plot_point_line.html

        :param lat:
        :param lon:
        :return:
        """

        import pyproj
        import math

        # Prepare data
        # - ensure that lat, lon are lists

        # Create projection system
        wgs = pyproj.Proj('epsg:4326')  # assuming you're using WGS84 geographic
        epsg = pyproj.CRS("+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs").format(
            lat=self.A1["lat"], lon=self.A1["lon"]
        ).to_epsg()
        crs = pyproj.Proj(epsg)

        # Transform points to projection system
        self.A1["x"], self.A1["y"] = pyproj.transform(wgs, crs, self.A1["lat"], self.A1["lon"])
        self.A2["x"], self.A2["y"] = pyproj.transform(wgs, crs, self.A2["lat"], self.A2["lon"])
        x, y = pyproj.transform(wgs, crs, lat, lon)
        # Note: The distance between self.A1["x", "y"] and self.A2["x", "y"] should be == self.length

        # Rotate points to line
        # - Project point to line
        # - Calculate distance to A1
        # - Remove points that are not within bounds (?)


        d = 0

        return d

