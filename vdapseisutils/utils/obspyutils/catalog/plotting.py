"""
Plotting functionality for VCatalog.

This module provides plotting methods for earthquake catalogs including
event rate plots and scatter plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from obspy.imaging.util import _set_xaxis_obspy_dates
from vdapseisutils.utils.magnitude import MagnitudeUtils


class VCatalogPlottingMixin:
    """Mixin providing plotting functionality for VCatalog."""
    
    def plot_eventrate(self, freq="1D", ax=None, **kwargs):
        """
        Plot a rate of earthquake events per time period.

        Parameters
        ----------
        freq : str, optional
            Frequency string for binning (default "1D" for daily)
            Examples: "1H" (hourly), "1D" (daily), "1W" (weekly), "1M" (monthly)
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure and axes
        **kwargs
            Additional keyword arguments passed to matplotlib step() function

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot
        """
        # Extract origin times from all events in the catalog
        origin_times = self.extract_origin_times()

        if not origin_times:
            raise ValueError("No events with origin times found in catalog")

        return VCatalogPlottingMixin.plot_eventrate_from_times(origin_times, freq, ax, **kwargs)

    @classmethod
    def plot_eventrate_from_catalog(cls, catalog, freq="1D", ax=None, **kwargs):
        """
        DEPRECATED: Use plot_eventrate_from_times() instead.
        
        This method is deprecated and will be removed in a future version.
        Use plot_eventrate_from_times() which automatically handles both
        catalogs and datetime lists.
        """
        import warnings
        warnings.warn(
            "plot_eventrate_from_catalog() is deprecated. "
            "Use plot_eventrate_from_times() instead, which automatically "
            "handles both catalogs and datetime lists.",
            DeprecationWarning,
            stacklevel=2
        )
        return VCatalogPlottingMixin.plot_eventrate_from_times(catalog, freq, ax, **kwargs)

    @staticmethod
    def plot_eventrate_from_times(times, freq="1D", ax=None, **kwargs):
        """
        Plot event rate from a list-like object of datetimes.

        Parameters
        ----------
        times : list-like
            List-like object of datetime objects, strings, or UTCDateTime objects
        freq : str, optional
            Frequency string for binning (default "1D" for daily)
            Examples: "1H" (hourly), "1D" (daily), "1W" (weekly), "1M" (monthly)
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure and axes
        **kwargs
            Additional keyword arguments passed to matplotlib step() function

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot
        """
        # Validate input
        if not hasattr(times, '__iter__') or isinstance(times, str):
            raise ValueError("Input must be a list-like object of datetimes")
        
        # Convert UTCDateTime objects to datetime if needed
        converted_times = []
        for time_obj in times:
            if hasattr(time_obj, 'datetime'):
                # ObsPy UTCDateTime object
                converted_times.append(time_obj.datetime)
            else:
                # Already a datetime-like object
                converted_times.append(time_obj)
        
        # Convert to pandas datetime index and resample to get counts per time period
        time_series = pd.Series(1, index=pd.DatetimeIndex(converted_times))
        counts = time_series.resample(freq).count()

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Set default color to black if not specified
        if 'color' not in kwargs:
            kwargs['color'] = 'black'

        # Create step plot
        ax.step(counts.index, counts.values, where='post', **kwargs)

        # Set up the x-axis with ObsPy date formatting
        _set_xaxis_obspy_dates(ax)

        # Set labels and limits
        ax.set_ylabel('Events per ' + freq)
        ax.set_xlabel('Time')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        return ax

    def scatter(self, ax=None, cmap='viridis', mscale=None, **kwargs):
        """
        Create a scatter plot of longitude (x) vs latitude (y) for catalog events.
        If available, marker size is based on magnitude and color is based on depth.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        cmap : str or Colormap, default 'viridis'
            Colormap for depth coloring.
        mscale : MagnitudeUtils instance, optional
            MagnitudeUtils instance to use for marker sizing. If None, uses default settings.
        **kwargs :
            Additional keyword arguments passed to plt.scatter.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the scatter plot.
        """
        lons = []
        lats = []
        mags = []
        depths = []
        for event in self:
            if event.origins:
                o = event.origins[0]
                lons.append(o.longitude if hasattr(o, 'longitude') and o.longitude is not None else np.nan)
                lats.append(o.latitude if hasattr(o, 'latitude') and o.latitude is not None else np.nan)
                depths.append(o.depth if hasattr(o, 'depth') and o.depth is not None else np.nan)
            else:
                lons.append(np.nan)
                lats.append(np.nan)
                depths.append(np.nan)
            if event.magnitudes:
                mags.append(event.magnitudes[0].mag if event.magnitudes[0].mag is not None else 1)
            else:
                mags.append(1)
        lons = np.array(lons)
        lats = np.array(lats)
        mags = np.array(mags)
        depths = np.array(depths)
        # Use MagnitudeUtils for marker size
        if mscale is None:
            mscale = MagnitudeUtils()
        sizes = mscale.to_size(mags)
        c = depths
        if ax is None:
            fig, ax = plt.subplots()
        sc = ax.scatter(lons, lats, s=sizes, c=c, cmap=cmap, alpha=kwargs.pop('alpha', 0.7), **kwargs)
        cb = plt.colorbar(sc, ax=ax, label='Depth (m)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Event Scatter Plot')
        return ax 