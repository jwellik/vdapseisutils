"""
TimeSeries class for creating time-series plots.

This module contains the TimeSeries class for creating time-series visualizations
of seismic data plotted against time.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import matplotlib.pyplot as plt
from obspy.imaging.util import _set_xaxis_obspy_dates

from .defaults import TICK_DEFAULTS, AXES_DEFAULTS, CROSSSECTION_DEFAULTS
from .utils import prep_catalog_data_mpl
from .legends import MagLegend
from vdapseisutils.utils.timeutils import convert_timeformat


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
        """Set the y-axis limits based on axis type."""
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
        """
        Create a scatter plot on the time-series.
        
        Parameters:
        -----------
        t : array-like
            Time values
        y : array-like
            Y-axis values (depth or magnitude)
        yaxis : str, optional
            Type of y-axis data (default: "Depth")
        **kwargs
            Additional scatter plot arguments
        """
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
        """
        Add a vertical line to the time-series plot.
        
        Parameters:
        -----------
        t : datetime-like
            Time value for the vertical line
        *args, **kwargs
            Additional arguments passed to matplotlib's axvline
        """
        self.ax.axvline(convert_timeformat(t, "matplotlib"), *args, **kwargs)


def _test_time_series():
    """Simple test to verify TimeSeries class works correctly."""
    try:
        # Test TimeSeries creation with default parameters (depth)
        ts_obj = TimeSeries(axis_type="depth", depth_extent=(-30, 5))
        
        print("✓ TimeSeries class created successfully (depth axis)")
        print(f"✓ Axis type: {ts_obj.axis_type}")
        print(f"✓ Depth extent: {ts_obj.depth_extent}")
        
        # Test TimeSeries with magnitude axis
        ts_obj2 = TimeSeries(axis_type="magnitude")
        print("✓ TimeSeries created with magnitude axis")
        
        # Test with colorbar
        ts_obj3 = TimeSeries(colorbar=True)
        print("✓ TimeSeries created with colorbar")
        
        return True
        
    except Exception as e:
        print(f"✗ TimeSeries test failed: {e}")
        return False


if __name__ == "__main__":
    _test_time_series()
