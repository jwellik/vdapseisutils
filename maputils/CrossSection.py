import matplotlib.pyplot as plt
import numpy as np
from vdapseisutils.maputils.utils import elev_profile

class CrossSection:

    def __init__(self, A1, A2,
                 n=100,
                 length='longitude',
                 depth_extent=(-50., 4.),  # -> float # km (bottom_depth, top_altitude)
                 figsize=(8, 4),
                 title="Cross Section A-A'",  # -> str
                 orientation='horizontal',
                 linewidth=0.75,
                 color='k',
                 ):

        self.points = [A1, A2]
        self.lat, self.lon, self.d, self.elev = elev_profile.download_profile2(self.points[0], self.points[1], n=n)  # elevation returned in meters
        self.elev = np.array(self.elev) / 1000  # convert to km
        self.length = length
        self.orientation = orientation
        self.depth_extent = depth_extent

        self.fig = plt.figure(figsize=figsize)

        if self.length == 'lon' or self.length == 'longitude':
            dist = self.lon
        elif self.length == 'lat' or self.length == 'latitude':
            dist = self.lat
        elif self.length == 'distance' or self.length == 'kilometers':
            dist = self.d


        if self.orientation == 'horizontal':
            plt.plot(dist, self.elev, color=color, linewidth=linewidth)
            self.fig.axes[0].spines['top'].set_visible(False)  # custom spine bounds for a nice clean look
            self.fig.axes[0].spines.left.set_bounds((self.depth_extent[0], self.elev[0]))  # depth_extent_v[1] is the top elev
            self.fig.axes[0].spines.right.set_bounds((self.depth_extent[0], self.elev[-1]))
            self.fig.axes[0].set_xlim((dist[0], dist[-1]))
            self.fig.axes[0].set_ylim(self.depth_extent)
        elif self.orientation == 'vertical':
            plt.plot(self.elev, dist, color=color, linewidth=linewidth)
            self.fig.axes[0].spines['left'].set_visible(False)  # custom spine bounds for a nice clean look
            self.fig.axes[0].spines.bottom.set_bounds((self.depth_extent[0], self.elev[0]))  # depth_extent_v[1] is the top elev
            self.fig.axes[0].spines.top.set_bounds((self.depth_extent[0], self.elev[-1]))
            self.fig.axes[0].set_ylim((dist[0], dist[-1]))
            self.fig.axes[0].set_xlim([self.depth_extent[1], self.depth_extent[0]])
            # move yticklabels to the right

        plt.draw()