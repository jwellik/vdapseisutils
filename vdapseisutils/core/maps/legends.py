"""
Legend classes for map visualizations.

This module contains legend-related classes including MagLegend for magnitude
scaling and future legend implementations.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import numpy as np
import matplotlib.pyplot as plt
from .defaults import AXES_DEFAULTS


class MagLegend:
    """
    Magnitude legend for scaling earthquake magnitudes to marker sizes.
    
    This class handles the conversion between earthquake magnitudes and
    matplotlib marker sizes/point sizes for consistent visualization
    across different plot types.
    """

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
        """Display the magnitude scaling relationship."""
        fig = plt.figure()
        plt.scatter(self.mrange, self.msrange, s=self.srange, color=color, alpha=alpha)
        plt.show()

    def display(self, ax=None, color="none", edgecolor="k", include_counts=True):
        """
        Display the magnitude legend on the specified axes.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw the legend on. If None, creates new figure and axes.
        color : str, optional
            Fill color for legend markers (default: "none")
        edgecolor : str, optional
            Edge color for legend markers (default: "k")
        include_counts : bool, optional
            Whether to include event counts in labels (default: True)
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the legend
        """
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
        """
        MAG2S Converts magnitude to point size for scatter plot
        
        Uses ranges set for magnitude and marksersize range
        Point size is markersize**2
        Default is M0 is roughly equal to default markersize of 6 (point size 36)
        
        Parameters:
        -----------
        mag : array-like
            Magnitude values to convert
            
        Returns:
        --------
        numpy.ndarray
            Point sizes for scatter plot (markersize**2)
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
        """Display information about the magnitude legend scaling."""
        print("::: Magnitude Legend Information :::")
        print("     ms: markersize (default=6)")
        print("     s : point size (markersize**2")
        for M, ms, s in zip(self.mrange, self.msrange, self.srange):
            print("M{:>-4.1f} | ms: {:>4.1f} | s: {:>4.1f}".format(M, ms, s))
        print()


class ColorBar:
    """Placeholder for future colorbar functionality."""
    pass


def _test_legends():
    """Simple test to verify legends module works correctly."""
    try:
        # Test MagLegend creation
        mag_legend = MagLegend()
        
        # Test mag2s conversion
        test_mags = [0, 1, 2, 3]
        sizes = mag_legend.mag2s(test_mags)
        
        print(f"✓ MagLegend created successfully")
        print(f"✓ Magnitude range: {mag_legend.mrange}")
        print(f"✓ Test magnitudes {test_mags} -> sizes {sizes}")
        
        # Test info method
        print("✓ MagLegend info:")
        mag_legend.info()
        
        return True
        
    except Exception as e:
        print(f"✗ Legends test failed: {e}")
        return False


if __name__ == "__main__":
    _test_legends()
