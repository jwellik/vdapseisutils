"""
Clipboard: Collection of Panels for multi-trace plotting
"""

# TODO If sync_waves=False and tick_type="absolute", increase panel_spacing and add xticks for each panel
# TODO Improve relative tick positions and labels (0 should be left and go from there)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from obspy import Stream, Trace, UTCDateTime
from .panel import Panel
from .timeaxes import TimeAxes


class Clipboard:
    """
    A collection of Panels for plotting multiple traces/streams.
    
    Each Panel typically represents one trace with waveform + spectrogram,
    but can be any combination of time-series plots. Handles synchronization
    of time axes across panels and manages tick formatting globally.
    """
    
    def __init__(self, data=None, sync_waves=False, figsize=(10, 12), 
                 panel_height=3, tick_type="absolute", wave_settings=None, 
                 spec_settings=None, panel_spacing=0.02, title_space=0.10, mode="wg"):
        """
        Initialize Clipboard.
        
        Parameters:
        -----------
        data : obspy.Stream, list of Traces, or None
            Data to plot. If provided, creates panels automatically.
        sync_waves : bool
            Whether to synchronize time axes across all panels. 
            Note: Within each panel, TimeAxes are always synchronized.
        figsize : tuple
            Figure size
        panel_height : float
            Height per panel in inches
        tick_type : str
            Type of tick formatting
        wave_settings : dict, optional
            Default settings for waveform plotting
        spec_settings : dict, optional
            Default settings for spectrogram plotting
        panel_spacing : float
            Spacing between panels (0.0 = no spacing, 0.1 = large spacing)
        title_space : float
            Fraction of figure height reserved for suptitle (0.10 = 10%)
        mode : str
            What to plot for each trace: "w" (waveform), "g" (spectrogram), "wg" (both, default)
        """
        self.panels = []
        self.sync_waves = sync_waves
        self.tick_type = tick_type
        self.panel_height = panel_height
        self.panel_spacing = panel_spacing
        self.title_space = title_space
        self.mode = mode
        self.wave_settings = wave_settings or {"color": "k"}
        self.spec_settings = spec_settings or {"wlen": 2.0, "overlap": 0.86, "dbscale": True}
        
        # Calculate figure size based on data
        if data is not None:
            if isinstance(data, Stream):
                n_panels = len(data)
            elif isinstance(data, (list, tuple)):
                n_panels = len(data)
            else:
                n_panels = 1
                data = [data]
        else:
            n_panels = 0
            
        if n_panels > 0:
            total_height = panel_height * n_panels
            self.fig = plt.figure(figsize=(figsize[0], total_height))
            self._create_panels_from_data(data)
        else:
            self.fig = plt.figure(figsize=figsize)
            
        self._time_extent = None  # Global time extent for sync_waves
        
    def _create_panels_from_data(self, data):
        """Create panels from input data."""
        if isinstance(data, Stream):
            traces = [tr for tr in data]
        elif isinstance(data, (list, tuple)):
            traces = data
        else:
            traces = [data]
            
        # Calculate global time extent if syncing
        if self.sync_waves:
            start_times = [tr.stats.starttime for tr in traces]
            end_times = [tr.stats.endtime for tr in traces]
            global_start = min(start_times)
            global_end = max(end_times)
            self._time_extent = (global_start, global_end)
            
        # Create a master gridspec that divides the figure into panel sections
        # Each panel gets a section, with spacing between panels but not within panels
        n_traces = len(traces)
        
        # Determine what to plot based on mode
        plot_waveform = "w" in self.mode
        plot_spectrogram = "g" in self.mode
        
        if not plot_waveform and not plot_spectrogram:
            raise ValueError(f"Invalid mode '{self.mode}'. Must contain 'w' and/or 'g'.")
        
        # Calculate relative heights based on mode
        if self.mode == "wg":
            height_ratios = [1, 3]  # waveform (1) + spectrogram (3)
            panel_height = 4
        elif self.mode == "w":
            height_ratios = [1]  # waveform only
            panel_height = 1
        elif self.mode == "g":
            height_ratios = [1]  # spectrogram only  
            panel_height = 1
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Use 'w', 'g', or 'wg'.")
            
        total_height = n_traces * panel_height
        
        # Create positions for each panel with spacing
        panel_positions = []
        spacing = self.panel_spacing  # Configurable space between panels
        
        # Reserve space at top for title
        title_space = self.title_space
        available_height = 1.0 - title_space
        
        for i in range(n_traces):
            # Calculate the top and bottom of this panel within available space
            panel_start = title_space + (i * (panel_height + spacing) / (total_height + (n_traces - 1) * spacing)) * available_height
            panel_end = title_space + ((i * (panel_height + spacing) + panel_height) / (total_height + (n_traces - 1) * spacing)) * available_height
            panel_positions.append((panel_start, panel_end))
        
        # Create panels
        for i, trace in enumerate(traces):
            # Create panel with its own gridspec in the allocated space
            panel = Panel(fig=self.fig, tick_type=self.tick_type)
            
            # Get the position for this panel
            panel_bottom, panel_top = panel_positions[n_traces - 1 - i]  # Reverse order (top to bottom)
            
            # Create gridspec for this panel based on mode
            n_subplots = len(height_ratios)
            panel_gs = self.fig.add_gridspec(n_subplots, 1, 
                                           height_ratios=height_ratios,
                                           hspace=0.0,  # No space within panel
                                           top=panel_top, 
                                           bottom=panel_bottom)
            
            # Add axes to panel based on mode
            axes_list = []
            for j in range(n_subplots):
                axes_list.append(self.fig.add_subplot(panel_gs[j]))
            
            # Create metadata for this trace
            trace_metadata = {
                'network': trace.stats.network,
                'station': trace.stats.station, 
                'location': trace.stats.location,
                'channel': trace.stats.channel,
                'id': trace.id,
                'nslc': f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}",
                'net_sta': f"{trace.stats.network}.{trace.stats.station}",
                'sta': trace.stats.station
            }
            
            # Create TimeAxes based on mode
            timeaxes_list = []
            axis_idx = 0
            
            if plot_waveform:
                wave_timeaxes = TimeAxes(ax=axes_list[axis_idx], tick_type=self.tick_type, metadata=trace_metadata.copy())
                wave_timeaxes.plot_waveform(trace, **self.wave_settings)
                wave_timeaxes.ax.set_xticks([])  # No x-ticks on waveform (unless it's the only plot)
                wave_timeaxes.ax.set_yticks([])  # No y-ticks on waveform
                
                # Set y-axis label for waveform if it's the only plot (mode="w")
                if self.mode == "w":
                    wave_timeaxes.set_ylabel(trace.id)
                
                timeaxes_list.append(wave_timeaxes)
                axis_idx += 1
            
            if plot_spectrogram:
                spec_timeaxes = TimeAxes(ax=axes_list[axis_idx], tick_type=self.tick_type, metadata=trace_metadata.copy())
                spec_timeaxes.plot_spectrogram(trace, **self.spec_settings)
                
                # Set y-axis label for spectrogram (always gets label in "g" and "wg" modes)
                spec_timeaxes.set_ylabel(trace.id)
                
                timeaxes_list.append(spec_timeaxes)
                axis_idx += 1
            
            panel.timeaxes = timeaxes_list
            panel._time_extent = (trace.stats.starttime, trace.stats.endtime)
            panel.metadata = trace_metadata.copy()  # Set panel metadata
            
            # Always sync TimeAxes within the panel to trace extent
            panel._sync_xlimits()
            
            # Handle x-axis labels - only on bottom panel
            if i < n_traces - 1:
                # Remove x-ticks from all axes except bottom panel
                for ta in timeaxes_list:
                    ta.ax.set_xticks([])
            else:
                # Bottom panel - only the last axes gets x-ticks
                if len(timeaxes_list) > 1:
                    for ta in timeaxes_list[:-1]:
                        ta.ax.set_xticks([])
                
            self.panels.append(panel)
            
        # Sync time limits if requested
        if self.sync_waves:
            self._sync_all_panels()
            
        # Format time axis on bottom panel
        if self.panels:
            self.panels[-1].set_tick_type(self.tick_type)
            
    def _sync_all_panels(self):
        """
        Synchronize time limits across all panels to global time extent.
        This is only called when sync_waves=True.
        Note: Within each panel, TimeAxes are always synchronized regardless of sync_waves.
        """
        if self._time_extent is None:
            return
            
        start_dt = self._time_extent[0].datetime
        end_dt = self._time_extent[1].datetime
        
        for panel in self.panels:
            panel.set_xlim(start_dt, end_dt)
            
    def add_panel(self, trace=None, height_ratios=[1, 3]):
        """
        Add a new panel to the clipboard.
        
        Parameters:
        -----------
        trace : obspy.Trace, optional
            Trace to plot in new panel. If None, creates empty panel.
        height_ratios : list
            Height ratios for axes in the panel
            
        Returns:
        --------
        Panel
            The newly created panel
        """
        # For now, this is a simplified version
        # In practice, you might want to rebuild the entire layout
        panel = Panel(fig=self.fig, height_ratios=height_ratios, tick_type=self.tick_type)
        
        if trace is not None:
            panel = Panel.from_trace_waveform_spectrogram(
                trace, height_ratios=height_ratios, 
                tick_type=self.tick_type,
                wave_settings=self.wave_settings,
                spec_settings=self.spec_settings
            )
            
        self.panels.append(panel)
        return panel
        
    def axvline(self, time_input, panels=None, axes=None, t_units="absolute", 
                stations=None, networks=None, ids=None, metadata=None, **kwargs):
        """
        Plot vertical line on specified panels/axes using numeric or metadata-based targeting.
        
        Parameters:
        -----------
        time_input : str, float, or datetime-like
            Time to plot line at
        panels : list, int, or None
            Which panels to plot on by index. If None, plots on all panels.
        axes : list, int, or None
            Which axes within panels to plot on. If None, plots on all axes in selected panels.
        t_units : str
            Units for time interpretation
        stations : str, list, or None
            Station codes to target (e.g., "ANMO" or ["ANMO", "CCM"])
        networks : str, list, or None
            Network codes to target (e.g., "IU" or ["IU", "US"])
        ids : str, list, or None
            Full trace IDs to target (e.g., "IU.ANMO.00.BHZ")
        metadata : dict, or None
            Custom metadata criteria to match against
        **kwargs : dict
            Additional arguments passed to axvline()
        """
        # Determine target panels using metadata or indices
        target_panels = self._resolve_target_panels(panels, stations, networks, ids, metadata)
            
        # Plot on selected panels
        for panel in target_panels:
            panel.axvline(time_input, axes=axes, t_units=t_units, **kwargs)
            
        return self
    
    def plot_trace(self, data, stations=None, networks=None, ids=None, metadata=None,
                   panels=None, axes=None, zorder=-1, **kwargs):
        """
        Plot additional traces on existing panels.
        
        This method allows you to overlay additional traces on existing panels,
        useful for plotting multiple components (N/E under Z) or comparing
        different time periods or processing levels.
        
        Parameters:
        -----------
        data : obspy.Trace or obspy.Stream
            The trace(s) to plot
        stations : list, optional
            Target panels by station codes (e.g., ['ANMO', 'CMB'])
        networks : list, optional
            Target panels by network codes (e.g., ['BK', 'CI'])
        ids : list, optional
            Target panels by full trace IDs (e.g., ['BK.CMB.00.BHZ'])
        metadata : dict, optional
            Target panels by custom metadata criteria
        panels : list, optional
            Target specific panel indices (e.g., [0, 2])
        axes : list, optional
            Target specific axes within panels (e.g., [0] for waveform, [1] for spectrogram)
        zorder : int
            Drawing order (lower values drawn first, -1 = behind existing plots)
        **kwargs
            Additional arguments passed to plot_waveform() (color, alpha, linewidth, etc.)
            Note: Do NOT pass Clipboard parameters like 'mode' or 'sync_waves' here
            
        Returns:
        --------
        Clipboard
            Self for method chaining
            
        Examples:
        ---------
        # Plot N/E components under Z components
        cb = swarmw(z_stream)
        cb.plot_trace(n_stream, color="red", alpha=0.5, zorder=-1)
        cb.plot_trace(e_stream, color="blue", alpha=0.5, zorder=-1)
        
        # Plot only on specific stations
        cb.plot_trace(stream, stations=["ANMO"], color="gray")
        
        # Plot only on waveform axes (not spectrograms)
        cb.plot_trace(stream, axes=[0], color="green")
        """
        from obspy import Stream, Trace
        
        # Convert single trace to stream for consistent handling
        if isinstance(data, Trace):
            data = Stream([data])
        elif not isinstance(data, Stream):
            raise TypeError("data must be an ObsPy Trace or Stream object")
        
        # Filter out invalid kwargs that don't belong in plot_waveform
        invalid_kwargs = ['mode', 'sync_waves', 'figsize', 'tick_type', 'panel_spacing', 'title_space']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in invalid_kwargs}
        
        if any(k in kwargs for k in invalid_kwargs):
            invalid_found = [k for k in invalid_kwargs if k in kwargs]
            print(f"⚠️  Ignoring invalid kwargs for plot_trace: {invalid_found}")
            print("   These parameters belong in the Clipboard constructor, not plot_trace()")
        
        # Resolve target panels
        target_panels = self._resolve_target_panels(panels, stations, networks, ids, metadata)
        
        # If no specific targeting, plot on all panels
        if not target_panels:
            target_panels = self.panels
        
        # Plot traces on target panels
        for trace in data:
            # Always match trace to panels by station metadata first
            trace_metadata = {
                'network': trace.stats.network,
                'station': trace.stats.station,
                'id': trace.id,
                'net_sta': f"{trace.stats.network}.{trace.stats.station}",
                'sta': trace.stats.station
            }
            
            # Find panels that match this trace's station
            matching_panels = []
            for panel in target_panels:
                if panel.matches_metadata(sta=trace.stats.station):
                    matching_panels.append(panel)
            
            # If no station matches found, skip this trace
            if not matching_panels:
                if panels is None and not any([stations, networks, ids, metadata]):
                    # Only warn if no specific targeting was requested
                    print(f"⚠️  No panel found for station {trace.stats.station} (trace: {trace.id})")
                continue
            
            trace_panels = matching_panels
            
            # Plot on matching panels
            for panel in trace_panels:
                # Determine which axes to plot on
                target_axes = axes if axes is not None else list(range(len(panel.timeaxes)))
                
                for ax_idx in target_axes:
                    if ax_idx < len(panel.timeaxes):
                        timeaxes = panel.timeaxes[ax_idx]
                        
                        # Plot the trace with specified zorder, preserving existing time axis
                        timeaxes.plot_waveform(trace, zorder=zorder, preserve_xlim=True, **filtered_kwargs)
        
        return self
    
    def plot_horizontals(self, stream, color="gray", alpha=0.7, zorder=-1, **kwargs):
        """
        Convenience method to plot horizontal components (N/E) under vertical (Z) components.
        
        This method automatically identifies N and E components in the provided stream
        and plots them on panels that match the corresponding Z component stations.
        
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
            Additional arguments passed to plot_waveform() (e.g., linewidth, linestyle)
            Note: Do NOT pass Clipboard parameters like 'mode' or 'sync_waves' here
            
        Returns:
        --------
        Clipboard
            Self for method chaining
            
        Examples:
        ---------
        # Plot Z components, then add N/E underneath
        cb = swarmw(z_stream)
        cb.plot_horizontals(full_stream)
        
        # Custom styling for horizontals
        cb.plot_horizontals(full_stream, color="red", alpha=0.3, linewidth=0.5)
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
        
        # Filter out invalid kwargs that don't belong in plot_waveform
        invalid_kwargs = ['mode', 'sync_waves', 'figsize', 'tick_type', 'panel_spacing', 'title_space']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in invalid_kwargs}
        
        if any(k in kwargs for k in invalid_kwargs):
            invalid_found = [k for k in invalid_kwargs if k in kwargs]
            print(f"⚠️  Ignoring invalid kwargs for plot_horizontals: {invalid_found}")
            print("   These parameters belong in the Clipboard constructor, not plot_horizontals()")
        
        # Plot horizontal components with specified styling
        self.plot_trace(horizontal_stream, color=color, alpha=alpha, zorder=zorder, **filtered_kwargs)
        
        return self
        
    def _resolve_target_panels(self, panels=None, stations=None, networks=None, ids=None, metadata=None):
        """
        Resolve target panels based on indices or metadata criteria.
        
        Returns:
        --------
        list
            List of Panel objects to target
        """
        # If numeric panels specified, use those
        if panels is not None:
            if isinstance(panels, int):
                return [self.panels[panels]]
            else:
                return [self.panels[i] for i in panels]
        
        # Otherwise, filter by metadata
        target_panels = []
        
        for panel in self.panels:
            # Check station criteria
            if stations is not None:
                station_list = stations if isinstance(stations, (list, tuple)) else [stations]
                if panel.metadata.get('sta') not in station_list and panel.metadata.get('station') not in station_list:
                    continue
                    
            # Check network criteria
            if networks is not None:
                network_list = networks if isinstance(networks, (list, tuple)) else [networks]
                if panel.metadata.get('network') not in network_list:
                    continue
                    
            # Check ID criteria
            if ids is not None:
                id_list = ids if isinstance(ids, (list, tuple)) else [ids]
                if panel.metadata.get('id') not in id_list:
                    continue
                    
            # Check custom metadata criteria
            if metadata is not None:
                if not panel.matches_metadata(**metadata):
                    continue
                    
            target_panels.append(panel)
        
        # If no metadata criteria specified, return all panels
        if stations is None and networks is None and ids is None and metadata is None:
            return self.panels
            
        return target_panels
    
    def plot_catalog(self, catalog, panels=None, axes=None, plot_picks=True, plot_origins=True,
                    origin_color="black", p_color="red", s_color="blue", verbose=False,
                    stations=None, networks=None, ids=None, metadata=None, **kwargs):
        """
        Plot catalog events and picks on specified panels/axes using numeric or metadata-based targeting.
        
        Parameters:
        -----------
        catalog : obspy.Catalog or obspy.Event
            ObsPy catalog containing events and picks, or single Event
        panels : list, int, or None
            Which panels to plot on by index. If None, plots on all panels.
        axes : list, int, or None
            Which axes within panels to plot on. If None, plots on all axes in selected panels.
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
        stations : str, list, or None
            Station codes to target (e.g., "ANMO" or ["ANMO", "CCM"])
        networks : str, list, or None
            Network codes to target (e.g., "IU" or ["IU", "US"])
        ids : str, list, or None
            Full trace IDs to target (e.g., "IU.ANMO.00.BHZ")
        metadata : dict, or None
            Custom metadata criteria to match against
        **kwargs : dict
            Additional arguments passed to axvline()
        """
        # Convert Event to Catalog if needed
        from obspy import Catalog
        from obspy.core.event.event import Event
        if isinstance(catalog, Event):
            catalog = Catalog([catalog])
        
        if verbose:
            print("Adding picks from catalog...")
            
        # Determine target panels using metadata or indices
        target_panels = self._resolve_target_panels(panels, stations, networks, ids, metadata)
        
        for event in catalog:
            # Plot origin time on all target panels
            if plot_origins and event.origins:
                origin_time = event.origins[0].time
                if verbose:
                    print(f" {origin_time} | Origin time")
                
                for panel in target_panels:
                    # Determine target axes within panel
                    if axes is None:
                        target_axes = panel.timeaxes
                    elif isinstance(axes, int):
                        target_axes = [panel.timeaxes[axes]]
                    else:
                        target_axes = [panel.timeaxes[i] for i in axes]
                    
                    for ta in target_axes:
                        ta.axvline(origin_time, color=origin_color, **kwargs)
            
            # Plot picks with station-specific targeting
            if plot_picks and event.picks:
                for pick in event.picks:
                    pick_station = pick.waveform_id.station_code
                    
                    if verbose:
                        print(f" {pick.time} | {pick_station:<5s} : {pick.phase_hint}")
                    
                    # Determine color based on phase
                    if pick.phase_hint and pick.phase_hint.upper() == "P":
                        color = p_color
                    elif pick.phase_hint and pick.phase_hint.upper() == "S":
                        color = s_color
                    else:
                        color = p_color  # Default to P color for unknown phases
                    
                    # Plot pick on panels that match the pick's station
                    for panel in target_panels:
                        # Check if this panel matches the pick's station
                        if (panel.metadata.get('station') == pick_station or 
                            panel.metadata.get('sta') == pick_station):
                            
                            # Determine target axes within panel
                            if axes is None:
                                target_axes = panel.timeaxes
                            elif isinstance(axes, int):
                                target_axes = [panel.timeaxes[axes]]
                            else:
                                target_axes = [panel.timeaxes[i] for i in axes]
                            
                            for ta in target_axes:
                                ta.axvline(pick.time, color=color, **kwargs)
        
        return self
        
    def set_tick_type(self, tick_type):
        """Set tick type for all panels."""
        self.tick_type = tick_type
        
        # Only set tick formatting on the bottom panel to avoid clutter
        for i, panel in enumerate(self.panels):
            if i == len(self.panels) - 1:  # Bottom panel
                panel.set_tick_type(tick_type)
            else:
                # Ensure no x-ticks on non-bottom panels
                for ta in panel.timeaxes:
                    ta.ax.set_xticks([])
                    
        return self
        
    def set_xlim(self, left=None, right=None):
        """Set x-limits for all panels."""
        for panel in self.panels:
            panel.set_xlim(left, right)
        return self

    def set_wlim(self, limits):
        """Sets ylim for waveform in all panels. Assumes waveform is bottom axes."""
        for panel in self.panels:
            panel.set_wlim(limits)
        return self

    def set_slim(self, limits):
        """Sets ylim for spectrogram in all panels. Assumes spectrogram is bottom axes."""
        for panel in self.panels:
            panel.set_slim(limits)
        return self

    def get_panel(self, index):
        """Get Panel by index."""
        return self.panels[index]
        
    def get_timeaxes(self, panel_index, axes_index):
        """Get specific TimeAxes by panel and axes index."""
        return self.panels[panel_index].timeaxes[axes_index]
        
    @property
    def all_axes(self):
        """Return all matplotlib axes across all panels."""
        axes = []
        for panel in self.panels:
            axes.extend(panel.axes)
        return axes
