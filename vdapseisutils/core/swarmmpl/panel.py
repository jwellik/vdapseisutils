"""
Panel: Collection of TimeAxes sharing a time axis
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from obspy import Trace, UTCDateTime
from .timeaxes import TimeAxes


class Panel:
    """
    A collection of TimeAxes that share the same time axis.
    
    Most commonly used for waveform + spectrogram plots, but flexible 
    enough to handle any combination of time-series plots.
    """
    
    def __init__(self, fig=None, figsize=(10, 6), height_ratios=None, 
                 hspace=0.0, tick_type="absolute", metadata=None):
        """
        Initialize Panel.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure, optional
            Figure to create panel in. If None, creates new figure.
        figsize : tuple
            Figure size if creating new figure
        height_ratios : list, optional
            Height ratios for subplots. If None, uses equal heights.
        hspace : float
            Horizontal spacing between subplots
        tick_type : str
            Type of tick formatting for time axis
        metadata : dict, optional
            Metadata dictionary for identifying this Panel (e.g., station info)
        """
        if fig is not None:
            self.fig = fig
        else:
            self.fig = plt.figure(figsize=figsize)
            
        self.timeaxes = []  # List of TimeAxes objects
        self.tick_type = tick_type
        self.height_ratios = height_ratios or []
        self.hspace = hspace
        self._gridspec = None
        self._time_extent = None
        self.metadata = metadata or {}  # Store metadata for identification
        
    def add_timeaxes(self, height_ratio=1):
        """
        Add a new TimeAxes to the panel.
        
        Parameters:
        -----------
        height_ratio : float
            Height ratio for this axes relative to others
            
        Returns:
        --------
        TimeAxes
            The newly created TimeAxes object
        """
        self.height_ratios.append(height_ratio)
        self._rebuild_layout()
        
        # Create new TimeAxes with the last axes in the figure, inheriting panel metadata
        new_timeaxes = TimeAxes(ax=self.fig.axes[-1], tick_type=self.tick_type, metadata=self.metadata.copy())
        self.timeaxes.append(new_timeaxes)
        
        # Always sync x-limits within panel if time extent is set
        if self._time_extent is not None:
            self._sync_xlimits()
        
        return new_timeaxes
        
    def _rebuild_layout(self):
        """Rebuild the gridspec layout when axes are added."""
        # Clear existing axes
        self.fig.clear()
        
        # Create new gridspec
        nrows = len(self.height_ratios)
        self._gridspec = self.fig.add_gridspec(nrows, 1, 
                                              height_ratios=self.height_ratios,
                                              hspace=self.hspace)
        
        # Add subplots
        for i in range(nrows):
            self.fig.add_subplot(self._gridspec[i])
            
        # Recreate TimeAxes objects if they exist
        if len(self.timeaxes) < nrows:
            # Add missing TimeAxes
            for i in range(len(self.timeaxes), nrows):
                self.timeaxes.append(TimeAxes(ax=self.fig.axes[i], tick_type=self.tick_type))
        else:
            # Update existing TimeAxes with new axes
            for i, ta in enumerate(self.timeaxes[:nrows]):
                ta.ax = self.fig.axes[i]
                ta.fig = self.fig
                
    @classmethod
    def from_trace_waveform_spectrogram(cls, trace, height_ratios=[1, 3], 
                                       figsize=(10, 6), tick_type="absolute",
                                       wave_settings=None, spec_settings=None):
        """
        Create a Panel with waveform and spectrogram from a single trace.
        
        Parameters:
        -----------
        trace : obspy.Trace
            The trace to plot
        height_ratios : list
            Height ratios [waveform, spectrogram]
        figsize : tuple
            Figure size
        tick_type : str
            Type of tick formatting
        wave_settings : dict, optional
            Settings for waveform plotting
        spec_settings : dict, optional  
            Settings for spectrogram plotting
            
        Returns:
        --------
        Panel
            Panel with waveform and spectrogram
        """
        if not isinstance(trace, Trace):
            raise TypeError("Input must be an ObsPy Trace object")
            
        # Default settings
        wave_settings = wave_settings or {"color": "k"}
        spec_settings = spec_settings or {"wlen": 2.0, "overlap": 0.86, "dbscale": True}
        
        # Create panel
        panel = cls(figsize=figsize, height_ratios=height_ratios, tick_type=tick_type)
        
        # Add waveform axes
        wave_axes = panel.add_timeaxes(height_ratios[0])
        wave_axes.plot_waveform(trace, **wave_settings)
        wave_axes.ax.set_xticks([])  # Remove x-ticks from top plot
        wave_axes.ax.set_yticks([])  # Remove y-ticks for cleaner look
        
        # Add spectrogram axes  
        spec_axes = panel.add_timeaxes(height_ratios[1])
        spec_axes.plot_spectrogram(trace, **spec_settings)
        
        # Set shared time extent
        panel._time_extent = (trace.stats.starttime, trace.stats.endtime)
        
        # Update panel metadata from trace
        panel._update_metadata_from_trace(trace)
        
        # Always sync x-limits within a panel
        panel._sync_xlimits()
        
        # Add trace ID as ylabel on spectrogram
        spec_axes.set_ylabel(trace.id)
        
        return panel
        
    def _sync_xlimits(self):
        """Synchronize x-limits across all TimeAxes in the panel."""
        if not self.timeaxes or self._time_extent is None:
            return
            
        start_dt = self._time_extent[0].datetime
        end_dt = self._time_extent[1].datetime
        
        for ta in self.timeaxes:
            ta.ax.set_xlim(start_dt, end_dt)
            
    def axvline(self, time_input, axes=None, t_units="absolute", **kwargs):
        """
        Plot vertical line on specified axes in the panel.
        
        Parameters:
        -----------
        time_input : str, float, or datetime-like
            Time to plot line at
        axes : list or int, optional
            Which axes to plot on. If None, plots on all axes.
            Can be list of indices or single index.
        t_units : str
            Units for time interpretation
        **kwargs : dict
            Additional arguments passed to axvline()
        """
        if axes is None:
            target_axes = self.timeaxes
        elif isinstance(axes, int):
            target_axes = [self.timeaxes[axes]]
        else:
            target_axes = [self.timeaxes[i] for i in axes]
            
        for ta in target_axes:
            ta.axvline(time_input, t_units=t_units, **kwargs)
            
        return self
        
    def set_tick_type(self, tick_type):
        """Set tick type for all TimeAxes in the panel."""
        self.tick_type = tick_type
        for ta in self.timeaxes:
            ta.set_tick_type(tick_type)
        self.timeaxes[0].set_xticks([])  # Top axes never has xticks
        return self
        
    def set_xlim(self, left=None, right=None):
        """Set x-limits for all TimeAxes in the panel."""
        for ta in self.timeaxes:
            ta.set_xlim(left, right)
            
        # Update internal time extent if both limits provided
        if left is not None and right is not None:
            from obspy import UTCDateTime
            if isinstance(left, str):
                left = UTCDateTime(left)
            elif hasattr(left, 'datetime'):
                left = UTCDateTime(left.datetime)
            elif hasattr(left, 'timestamp'):
                left = UTCDateTime(left)
                
            if isinstance(right, str):
                right = UTCDateTime(right)
            elif hasattr(right, 'datetime'):
                right = UTCDateTime(right.datetime)  
            elif hasattr(right, 'timestamp'):
                right = UTCDateTime(right)
                
            self._time_extent = (left, right)
            
        return self
        
    def set_slim(self, limits):
        """Set ylimits for spectrograms. Assumes spectrogram is bottom axes"""
        self.timeaxes[-1].set_ylim(limits)

    def set_wlim(self, limits):
        """Set ylimits for spectrograms. Assumes spectrogram is top axes"""
        self.timeaxes[0].set_ylim(limits)

    def get_timeaxes(self, index):
        """Get TimeAxes by index."""
        return self.timeaxes[index]
        
    def _update_metadata_from_trace(self, trace):
        """Auto-populate panel metadata from ObsPy trace stats."""
        # Only update if metadata fields don't already exist
        if 'network' not in self.metadata:
            self.metadata['network'] = trace.stats.network
        if 'station' not in self.metadata:
            self.metadata['station'] = trace.stats.station
        if 'location' not in self.metadata:
            self.metadata['location'] = trace.stats.location
        if 'channel' not in self.metadata:
            self.metadata['channel'] = trace.stats.channel
        if 'id' not in self.metadata:
            self.metadata['id'] = trace.id
        if 'nslc' not in self.metadata:
            self.metadata['nslc'] = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}"
        if 'net_sta' not in self.metadata:
            self.metadata['net_sta'] = f"{trace.stats.network}.{trace.stats.station}"
        if 'sta' not in self.metadata:
            self.metadata['sta'] = trace.stats.station
            
        # Update all TimeAxes in this panel with the same metadata
        for ta in self.timeaxes:
            ta.metadata.update(self.metadata)
    
    def matches_metadata(self, **criteria):
        """
        Check if this Panel matches the given metadata criteria.
        
        Parameters:
        -----------
        **criteria : dict
            Metadata criteria to match against
            
        Returns:
        --------
        bool
            True if all criteria match
        """
        for key, value in criteria.items():
            if key not in self.metadata:
                return False
            if isinstance(value, (list, tuple)):
                if self.metadata[key] not in value:
                    return False
            else:
                if self.metadata[key] != value:
                    return False
        return True
    
    def plot_catalog(self, catalog, axes=None, plot_picks=True, plot_origins=True, 
                    origin_color="black", p_color="red", s_color="blue", 
                    verbose=False, **kwargs):
        """
        Plot catalog events and picks on specified axes in this Panel.
        
        Parameters:
        -----------
        catalog : obspy.Catalog or obspy.Event
            ObsPy catalog containing events and picks, or single Event
        axes : list, int, or None
            Which axes to plot on. If None, plots on all axes.
        plot_picks : bool
            Whether to plot picks
        plot_origins : bool
            Whether to plot origin times
        origin_color : str
            Color for origin time lines
        p_color : str
            Color for P-phase picks
        s_color : str
            Color for S-phase picks
        verbose : bool
            Whether to print information about plotted events
        **kwargs : dict
            Additional arguments passed to axvline()
        """
        # Convert Event to Catalog if needed
        from obspy import Catalog, Event
        if isinstance(catalog, Event):
            catalog = Catalog([catalog])
        
        # Determine target axes
        if axes is None:
            target_axes = self.timeaxes
        elif isinstance(axes, int):
            target_axes = [self.timeaxes[axes]]
        else:
            target_axes = [self.timeaxes[i] for i in axes]
        
        # Plot catalog on each target axes
        for ta in target_axes:
            ta.plot_catalog(catalog, plot_picks=plot_picks, plot_origins=plot_origins,
                           origin_color=origin_color, p_color=p_color, s_color=s_color,
                           verbose=verbose, **kwargs)
        
        return self
    
    def plot_trace(self, data, axes=None, zorder=-1, **kwargs):
        """
        Plot additional traces on this panel.
        
        Parameters:
        -----------
        data : obspy.Trace or obspy.Stream
            The trace(s) to plot
        axes : list, optional
            Target specific axes within panel (e.g., [0] for waveform, [1] for spectrogram)
        zorder : int
            Drawing order (lower values drawn first, -1 = behind existing plots)
        **kwargs
            Additional arguments passed to plot_waveform() (color, alpha, etc.)
            
        Returns:
        --------
        Panel
            Self for method chaining
        """
        from obspy import Stream, Trace
        
        # Convert single trace to stream for consistent handling
        if isinstance(data, Trace):
            data = Stream([data])
        elif not isinstance(data, Stream):
            raise TypeError("data must be an ObsPy Trace or Stream object")
        
        # Determine which axes to plot on
        target_axes = axes if axes is not None else list(range(len(self.timeaxes)))
        
        # Plot traces on target axes
        for trace in data:
            for ax_idx in target_axes:
                if ax_idx < len(self.timeaxes):
                    timeaxes = self.timeaxes[ax_idx]
                    
                    # Plot the trace with specified zorder, preserving existing time axis
                    timeaxes.plot_waveform(trace, zorder=zorder, preserve_xlim=True, **kwargs)
        
        return self
    
    def plot_horizontals(self, stream, color="gray", alpha=0.7, zorder=-1, **kwargs):
        """
        Convenience method to plot horizontal components (N/E) under vertical (Z) components.
        
        Parameters:
        -----------
        stream : obspy.Stream
            Stream containing horizontal components (channels ending with N or E)
        color : str
            Color for horizontal components (default: "gray")
        alpha : float
            Transparency for horizontal components (default: 0.7)
        zorder : int
            Drawing order (default: -1 = behind existing plots)
        **kwargs
            Additional arguments passed to plot_waveform()
            
        Returns:
        --------
        Panel
            Self for method chaining
        """
        from obspy import Stream
        
        if not isinstance(stream, Stream):
            raise TypeError("stream must be an ObsPy Stream object")
        
        # Extract horizontal components (N and E channels)
        horizontal_stream = Stream()
        
        for trace in stream:
            channel = trace.stats.channel
            if len(channel) >= 3 and channel[-1].upper() in ['N', 'E']:
                horizontal_stream.append(trace)
        
        if len(horizontal_stream) == 0:
            print("⚠️  No horizontal components (N/E) found in stream")
            return self
        
        # Plot horizontal components with specified styling
        self.plot_trace(horizontal_stream, color=color, alpha=alpha, zorder=zorder, **kwargs)
        
        return self
    
    @property
    def axes(self):
        """Return list of matplotlib axes for backward compatibility."""
        return [ta.ax for ta in self.timeaxes]
