"""
TimeAxes: Time-aware axes wrapper for seismic data plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from obspy import UTCDateTime, Trace
from obspy.imaging.util import _set_xaxis_obspy_dates
from matplotlib.dates import num2date, date2num

# Import vdap_colors if available, otherwise use matplotlib defaults
try:
    from vdapseisutils.style import colors as vdap_colors
except ImportError:
    vdap_colors = None

# Import timeutils if available
try:
    from vdapseisutils.utils.timeutils import convert_timeformat
except ImportError:
    convert_timeformat = None


class TimeAxes:
    """
    A time-aware wrapper around matplotlib Axes for plotting seismic timeseries data.
    
    Handles time-based x-axis formatting and provides time-aware plotting methods.
    The underlying x-values are always datetime objects, but tick labels can be 
    formatted as absolute times or relative times in various units.
    """
    
    def __init__(self, ax=None, fig=None, tick_type="absolute", figsize=(10, 6), metadata=None):
        """
        Initialize TimeAxes.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to wrap. If None, creates new figure and axes.
        fig : matplotlib.figure.Figure, optional
            Figure to create axes in. Only used if ax is None.
        tick_type : str
            Type of tick formatting. Options: "absolute", "relative", "seconds", 
            "minutes", "hours", "days", "months", "years"
        figsize : tuple
            Figure size if creating new figure
        metadata : dict, optional
            Metadata dictionary for identifying this TimeAxes (e.g., station info)
        """
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        elif fig is not None:
            self.fig = fig
            self.ax = fig.add_subplot(111)
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
            
        self.tick_type = tick_type
        self._time_extent = None  # Will store (start_time, end_time) as UTCDateTime
        self.metadata = metadata or {}  # Store metadata for identification
        
    def plot_waveform(self, trace, color="k", preserve_xlim=False, **kwargs):
        """
        Plot waveform of an ObsPy Trace.
        
        Parameters:
        -----------
        trace : obspy.Trace
            The trace to plot
        color : str
            Color for the waveform
        preserve_xlim : bool
            If True, preserve existing xlim instead of setting to trace extent
        **kwargs : dict
            Additional arguments passed to ax.plot()
        """
        if not isinstance(trace, Trace):
            raise TypeError("Input must be an ObsPy Trace object")
            
        # Convert times to datetime objects for plotting
        times = [trace.stats.starttime.datetime + timedelta(seconds=t) for t in trace.times()]
        
        # Store time extent
        self._time_extent = (trace.stats.starttime, trace.stats.endtime)
        
        # Auto-populate metadata from trace if not already set
        self._update_metadata_from_trace(trace)
        
        # Plot the waveform
        self.ax.plot(times, trace.data, color=color, **kwargs)
        
        # Set xlim to data extent (unless preserving existing xlim)
        if not preserve_xlim:
            self.ax.set_xlim(times[0], times[-1])
        
        # Configure axes
        self.ax.yaxis.set_ticks_position("right")
        self.ax.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        
        # Set time formatting
        self._format_time_axis()
        
        return self
        
    def plot_spectrogram(self, trace, samp_rate=None, wlen=2.0, overlap=0.86, 
                        dbscale=True, log_power=False, cmap=None, **kwargs):
        """
        Plot spectrogram of an ObsPy Trace.
        
        Parameters:
        -----------
        trace : obspy.Trace
            The trace to plot
        samp_rate : float, optional
            Resampling rate. If None, uses trace's sampling rate
        wlen : float
            Window length in seconds
        overlap : float
            Overlap fraction (0-1)
        dbscale : bool
            Whether to use dB scaling
        log_power : bool
            Whether to use log scale for frequency axis
        cmap : str or matplotlib colormap
            Colormap for spectrogram
        **kwargs : dict
            Additional arguments
        """
        from scipy.signal import spectrogram
        from obspy.imaging.spectrogram import _nearest_pow_2
        
        if not isinstance(trace, Trace):
            raise TypeError("Input must be an ObsPy Trace object")
            
        if cmap is None:
            if vdap_colors is not None:
                cmap = vdap_colors.inferno_u
            else:
                cmap = "inferno"  # matplotlib default
            
        # Resample if needed
        tr_work = trace.copy()
        if samp_rate:
            tr_work.resample(float(samp_rate))
        else:
            samp_rate = tr_work.stats.sampling_rate
            
        # Calculate spectrogram
        fs = tr_work.stats.sampling_rate
        signal = tr_work.data
        
        nfft = int(_nearest_pow_2(wlen * samp_rate))
        
        if len(signal) < nfft:
            raise ValueError(f'Input signal too short ({len(signal)} samples, '
                           f'window length {wlen} seconds, nfft {nfft} samples)')
                           
        nlap = int(nfft * float(overlap))
        signal = signal - signal.mean()
        
        frequencies, times_spec, Sxx = spectrogram(signal, fs=fs, nperseg=nfft, 
                                                  noverlap=nlap, scaling='spectrum')
        
        # Process power values
        if dbscale:
            Sxx = 10 * np.log10(Sxx[1:, :])
        else:
            Sxx = np.sqrt(Sxx[1:, :])
        frequencies = frequencies[1:]
        
        # Convert times to datetime objects
        start_date = tr_work.stats.starttime.datetime
        times_plot = [start_date + timedelta(seconds=t) for t in times_spec]
        
        # Store time extent
        self._time_extent = (tr_work.stats.starttime, tr_work.stats.endtime)
        
        # Auto-populate metadata from trace if not already set
        self._update_metadata_from_trace(trace)
        
        # Plot spectrogram
        self.ax.pcolormesh(times_plot, frequencies, Sxx, shading='auto', cmap=cmap, **kwargs)
        
        # Set xlim to data extent
        self.ax.set_xlim(times_plot[0], times_plot[-1])
        
        # Configure axes
        if log_power:
            self.ax.set_yscale('log')
        self.ax.set_ylim(0.1, samp_rate / 2.0)
        self.ax.yaxis.set_ticks_position("right")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        
        # Set time formatting
        self._format_time_axis()
        
        # Store spectrogram data for potential later use
        self._spec_data = {"freq": frequencies, "times": times_plot, "power": Sxx}
        
        return self
        
    def axvline(self, time_input, t_units="absolute", **kwargs):
        """
        Plot a vertical line at a specified time.
        
        Parameters:
        -----------
        time_input : str, float, or datetime-like
            Time to plot line at. If string, parsed as absolute time.
            If float, interpreted according to t_units.
        t_units : str
            Units for float input: "absolute", "seconds", "minutes", "hours"
        **kwargs : dict
            Additional arguments passed to ax.axvline()
        """
        if t_units == "absolute" or isinstance(time_input, (str, datetime)):
            # Parse as absolute time
            if isinstance(time_input, str):
                time_dt = UTCDateTime(time_input).datetime
            elif isinstance(time_input, datetime):
                time_dt = time_input
            else:
                time_dt = UTCDateTime(time_input).datetime
        else:
            # Parse as relative time
            if self._time_extent is None:
                raise ValueError("Cannot use relative time without plotting data first")
                
            start_time = self._time_extent[0].datetime
            
            if t_units == "seconds":
                time_dt = start_time + timedelta(seconds=float(time_input))
            elif t_units == "minutes":
                time_dt = start_time + timedelta(minutes=float(time_input))
            elif t_units == "hours":
                time_dt = start_time + timedelta(hours=float(time_input))
            else:
                raise ValueError(f"Unknown t_units: {t_units}")
                
        # Plot the vertical line
        self.ax.axvline(time_dt, **kwargs)
        return self
        
    def set_tick_type(self, tick_type):
        """
        Set the tick formatting type.
        
        Parameters:
        -----------
        tick_type : str
            "absolute", "relative", "seconds", "minutes", "hours", "days", "months", "years"
        """
        self.tick_type = tick_type
        self._format_time_axis()
        return self
        
    def _format_time_axis(self):
        """Format the time axis based on current tick_type."""
        if self._time_extent is None:
            return  # No data plotted yet
            
        if self.tick_type == "absolute":
            _set_xaxis_obspy_dates(self.ax)
            return
            
        # For relative time formatting
        _set_xaxis_obspy_dates(self.ax)  # Ensure datetime ticks first
        
        xticks = self.ax.get_xticks()
        if len(xticks) == 0:
            return
            
        xtick_dt = [num2date(x) for x in xticks]
        start_time = xtick_dt[0]
        
        # Calculate duration and determine appropriate intervals
        duration_seconds = (xtick_dt[-1] - start_time).total_seconds()
        
        if self.tick_type in ["relative", "seconds"]:
            self._set_relative_ticks(start_time, duration_seconds, "seconds")
        elif self.tick_type == "minutes":
            self._set_relative_ticks(start_time, duration_seconds, "minutes")  
        elif self.tick_type == "hours":
            self._set_relative_ticks(start_time, duration_seconds, "hours")
        elif self.tick_type == "days":
            self._set_relative_ticks(start_time, duration_seconds, "days")
        # TODO: Add months and years formatting
            
    def _set_relative_ticks(self, start_time, duration_seconds, unit):
        """Set relative time ticks for the specified unit."""
        if unit == "seconds":
            # Determine tick interval based on duration
            if duration_seconds <= 120:  # <= 2 minutes
                tick_interval = 30
            elif duration_seconds <= 600:  # <= 10 minutes
                tick_interval = 60
            else:
                tick_interval = 120
                
            max_ticks = int(duration_seconds / tick_interval) + 1
            nice_values = [i * tick_interval for i in range(max_ticks)]
            nice_tick_dt = [start_time + timedelta(seconds=s) for s in nice_values]
            nice_tick_labels = [f"{s}" for s in nice_values]
            nice_tick_labels[-1] += " s"
            
        elif unit == "minutes":
            duration_min = duration_seconds / 60
            if duration_min <= 10:
                tick_interval = 2
            elif duration_min <= 60:
                tick_interval = 10
            else:
                tick_interval = 30
                
            max_ticks = int(duration_min / tick_interval) + 1
            nice_values = [i * tick_interval for i in range(max_ticks)]
            nice_tick_dt = [start_time + timedelta(minutes=m) for m in nice_values]
            nice_tick_labels = [f"{m}" for m in nice_values]
            nice_tick_labels[-1] += " min"
            
        elif unit == "hours":
            duration_hr = duration_seconds / 3600
            if duration_hr <= 6:
                tick_interval = 1
            elif duration_hr <= 24:
                tick_interval = 4
            else:
                tick_interval = 12
                
            max_ticks = int(duration_hr / tick_interval) + 1
            nice_values = [i * tick_interval for i in range(max_ticks)]
            nice_tick_dt = [start_time + timedelta(hours=h) for h in nice_values]
            nice_tick_labels = [f"{h}" for h in nice_values]
            nice_tick_labels[-1] += " hr"
            
        elif unit == "days":
            duration_days = duration_seconds / 86400
            if duration_days <= 7:
                tick_interval = 1
            elif duration_days <= 30:
                tick_interval = 7
            else:
                tick_interval = 30
                
            max_ticks = int(duration_days / tick_interval) + 1
            nice_values = [i * tick_interval for i in range(max_ticks)]
            nice_tick_dt = [start_time + timedelta(days=d) for d in nice_values]
            nice_tick_labels = [f"{d}" for d in nice_values]
            nice_tick_labels[-1] += " days"
        
        # Apply the new ticks
        xticks_new = [date2num(x) for x in nice_tick_dt]
        self.ax.set_xticks(xticks_new)
        self.ax.set_xticklabels(nice_tick_labels)
        
    def set_xlim(self, left=None, right=None):
        """Set x-axis limits, accepting time strings or datetime objects."""
        if left is not None:
            if isinstance(left, str):
                left = UTCDateTime(left).datetime
            elif hasattr(left, 'datetime'):  # UTCDateTime
                left = left.datetime
                
        if right is not None:
            if isinstance(right, str):
                right = UTCDateTime(right).datetime
            elif hasattr(right, 'datetime'):  # UTCDateTime
                right = right.datetime
                
        self.ax.set_xlim(left, right)
        self._format_time_axis()  # Reformat after changing limits
        return self
        
    def set_ylim(self, bottom=None, top=None):
        """Set y-axis limits."""
        self.ax.set_ylim(bottom, top)
        return self
        
    def set_ylabel(self, label, **kwargs):
        """Set y-axis label."""
        self.ax.set_ylabel(label, **kwargs)
        return self
        
    def set_xlabel(self, label, **kwargs):
        """Set x-axis label."""
        self.ax.set_xlabel(label, **kwargs)
        return self
        
    def _update_metadata_from_trace(self, trace):
        """Auto-populate metadata from ObsPy trace stats."""
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
    
    def matches_metadata(self, **criteria):
        """
        Check if this TimeAxes matches the given metadata criteria.
        
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
    
    def plot_catalog(self, catalog, plot_picks=True, plot_origins=True, 
                    origin_color="black", p_color="red", s_color="blue", 
                    verbose=False, **kwargs):
        """
        Plot catalog events and picks on this TimeAxes.
        
        Parameters:
        -----------
        catalog : obspy.Catalog or obspy.Event
            ObsPy catalog containing events and picks, or single Event
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
        
        if verbose:
            print("Adding picks from catalog...")
            
        for event in catalog:
            # Plot origin time
            if plot_origins and event.origins:
                origin_time = event.origins[0].time
                if verbose:
                    print(f" {origin_time} | Origin time")
                self.axvline(origin_time, color=origin_color, **kwargs)
            
            # Plot picks
            if plot_picks and event.picks:
                for pick in event.picks:
                    # Check if this pick matches our metadata (station)
                    pick_station = pick.waveform_id.station_code
                    if ('station' in self.metadata and 
                        self.metadata['station'] != pick_station):
                        continue  # Skip picks not for this station
                        
                    if verbose:
                        print(f" {pick.time} | {pick_station:<5s} : {pick.phase_hint}")
                    
                    # Determine color based on phase
                    if pick.phase_hint and pick.phase_hint.upper() == "P":
                        color = p_color
                    elif pick.phase_hint and pick.phase_hint.upper() == "S":
                        color = s_color
                    else:
                        color = p_color  # Default to P color for unknown phases
                        
                    self.axvline(pick.time, color=color, **kwargs)
        
        return self
    
    def plot_trace(self, data, zorder=-1, **kwargs):
        """
        Plot additional traces on this TimeAxes.
        
        Parameters:
        -----------
        data : obspy.Trace or obspy.Stream
            The trace(s) to plot
        zorder : int
            Drawing order (lower values drawn first, -1 = behind existing plots)
        **kwargs
            Additional arguments passed to plot_waveform() (color, alpha, etc.)
            
        Returns:
        --------
        TimeAxes
            Self for method chaining
        """
        from obspy import Stream, Trace
        
        # Convert single trace to stream for consistent handling
        if isinstance(data, Trace):
            data = Stream([data])
        elif not isinstance(data, Stream):
            raise TypeError("data must be an ObsPy Trace or Stream object")
        
        # Plot traces on this axes, preserving existing time axis
        for trace in data:
            self.plot_waveform(trace, zorder=zorder, preserve_xlim=True, **kwargs)
        
        return self
    
    # Delegate common matplotlib methods to the wrapped axes
    def __getattr__(self, name):
        """Delegate unknown methods to the wrapped axes."""
        return getattr(self.ax, name)
