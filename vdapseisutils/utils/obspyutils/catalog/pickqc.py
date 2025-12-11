"""
Pick quality control functionality for VCatalog.

This module provides methods for comparing phase arrivals between catalogs
and visualizing the results for quality control purposes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class VCatalogPickQCMixin:
    """Mixin providing pick quality control functionality for VCatalog."""
    
    @staticmethod
    def get_pick_deltas(cat1, cat2, threshold_seconds=5, verbose=False):
        """
        Compare phase arrivals between two catalogs and calculate time differences.

        This method compares phase arrivals between corresponding events in two catalogs.
        It first finds matching events using origin time comparison, then for each matched
        event pair, it finds phase arrivals that match by station code, component
        (last character of channel: Z, N, E), and phase hint, then calculates the
        time difference (cat1 - cat2).

        Parameters
        ----------
        cat1 : obspy.core.event.Catalog or obspy.core.event.Event
            First catalog for comparison. If Event, will be converted to single-event Catalog
        cat2 : obspy.core.event.Catalog or obspy.core.event.Event
            Second catalog for comparison. If Event, will be converted to single-event Catalog
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds for matching events by origin time
        verbose : bool, default False
            If True, print detailed information about the matching process

        Returns
        -------
        pandas.DataFrame
            DataFrame containing comparison results with columns:
            - idx: Event pair index from matching results (0-based)
            - station: Station code
            - channel: Channel component (Z, N, or E)
            - phase_hint: Phase hint (P, S, etc.)
            - time_cat1: Pick time from cat1
            - time_cat2: Pick time from cat2
            - delta_t: Time difference (cat1 - cat2) in seconds
            - delta_abs_t: Absolute value of delta_t in seconds

        Examples
        --------
        >>> df = VCatalogPickQCMixin.get_pick_deltas(manual_catalog, automatic_catalog)
        >>> print(df[['station', 'phase_hint', 'delta_t']].head())

        >>> # With verbose output
        >>> df = VCatalogPickQCMixin.get_pick_deltas(cat1, cat2, verbose=True)
        """
        import pandas as pd
        from obspy.core.event import Catalog, Event

        if verbose:
            print("Starting pick comparison...")
            print(f"Catalog 1: {len(cat1)} events")
            print(f"Catalog 2: {len(cat2)} events")

        # Convert single events to catalogs
        if isinstance(cat1, Event):
            cat1 = Catalog([cat1])
        if isinstance(cat2, Event):
            cat2 = Catalog([cat2])

        # Find matching events by origin time
        from .comparison import VCatalogComparisonMixin
        matching_cat, idx1, idx2 = VCatalogComparisonMixin.compare_catalogs(
            cat1, cat2, threshold_seconds=threshold_seconds)
        
        if verbose:
            print(f"Found {len(idx1)} matching event pairs out of {len(cat1)} and {len(cat2)} events")

        # Collect all pick comparisons
        all_comparisons = []
        
        for pair_idx, (i1, i2) in enumerate(zip(idx1, idx2)):
            event1 = cat1[i1]
            event2 = cat2[i2]
            if verbose:
                print(f"\nProcessing event pair {pair_idx + 1}/{len(idx1)} (indices {i1}, {i2})")
                print(f"  Event 1 origin time: {event1.preferred_origin().time if event1.preferred_origin() else 'No origin'}")
                print(f"  Event 2 origin time: {event2.preferred_origin().time if event2.preferred_origin() else 'No origin'}")
                if event1.preferred_origin() and event2.preferred_origin():
                    time_diff = float(event1.preferred_origin().time - event2.preferred_origin().time)
                    print(f"  Time difference: {time_diff:.3f} seconds")
            
            # Get picks from both events
            picks1 = event1.picks if event1.picks else []
            picks2 = event2.picks if event2.picks else []
            
            if verbose:
                print(f"  Event 1 has {len(picks1)} picks")
                print(f"  Event 2 has {len(picks2)} picks")
            
            # Create dictionaries for efficient lookup
            # Key format: (station_code, component, phase_hint)
            if verbose:
                print("Picks from cat1")
            picks1_dict = {}
            for pick in picks1:
                if pick.waveform_id and pick.waveform_id.station_code:
                    station = pick.waveform_id.station_code
                    # Extract component (last character of channel)
                    component = pick.waveform_id.channel_code[-1] if pick.waveform_id.channel_code else None
                    phase_hint = pick.phase_hint if pick.phase_hint else None

                    if verbose:
                        print(f"  Pick: {station:<5s} {component} {phase_hint} {pick.time}")
                    
                    key = (station, component, phase_hint)
                    picks1_dict[key] = pick
            
            if verbose:
                print("\nPicks from cat2")
            picks2_dict = {}
            for pick in picks2:
                if pick.waveform_id and pick.waveform_id.station_code:
                    station = pick.waveform_id.station_code
                    component = pick.waveform_id.channel_code[-1] if pick.waveform_id.channel_code else None
                    phase_hint = pick.phase_hint if pick.phase_hint else None

                    if verbose:
                        print(f"  Pick: {station:<5s} {component} {phase_hint} {pick.time}")
                    
                    key = (station, component, phase_hint)
                    picks2_dict[key] = pick
            
            # Find matching picks and calculate differences
            matches_found = 0
            for key in picks1_dict:
                if key in picks2_dict:
                    pick1 = picks1_dict[key]
                    pick2 = picks2_dict[key]
                    
                    station, component, phase_hint = key
                    
                    # Calculate time difference (cat1 - cat2)
                    time1 = pick1.time
                    time2 = pick2.time
                    delta_t = float(time1 - time2)
                    
                    comparison = {
                        'idx': pair_idx,
                        'station': station,
                        'channel': component,
                        'phase_hint': phase_hint,
                        'time_cat1': time1,
                        'time_cat2': time2,
                        'delta_t': delta_t,
                        'delta_abs_t': abs(delta_t)
                    }
                    
                    all_comparisons.append(comparison)
                    matches_found += 1
            
            if verbose:
                print(f"  Found {matches_found} matching picks")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_comparisons)
        
        if verbose:
            print(f"\nSummary:")
            print(f"Total pick comparisons: {len(df)}")
            if len(df) > 0:
                print(f"Mean absolute difference: {df['delta_abs_t'].mean():.3f} seconds")
                print(f"Std absolute difference: {df['delta_abs_t'].std():.3f} seconds")
                print(f"Stations involved: {sorted(df['station'].unique())}")
                print(f"Phase types: {sorted(df['phase_hint'].unique())}")
        
        return df

    @staticmethod
    def plot_pick_deltas_histogram(df, figsize=(20, None), sort_by=None, include_phases=['P', 'S']):
        """
        Plot histogram of pick time deltas for various stations.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame from get_pick_deltas() containing
            columns: station, phase_hint, delta_t
        figsize : tuple, optional
            Figure size as (width, height). If height is None, it will be 
            calculated based on number of stations (default: (20, None))
        sort_by : str, list, or None, optional
            How to sort stations. Options:
            - None: alphabetical order (default)
            - 'total_picks': sort by total number of picks (ascending)
            - 'total_p': sort by number of P picks (ascending) 
            - 'total_s': sort by number of S picks (ascending)
            - 'best': sort by overall timing precision (highest combined std first, descending)
            - 'best_p': sort by P timing precision (highest P std first, descending)
            - 'best_s': sort by S timing precision (highest S std first, descending)
            - list: custom order of station codes
        include_phases : list, optional
            List of phase types to include in the plot (default: ['P', 'S'])
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        axes : numpy.ndarray
            Array of axes objects
        """
        # Group data by station and determine station order
        grouped = df.groupby('station')
        
        # Determine station ordering
        if sort_by is None:
            # Default: alphabetical order
            station_order = sorted(grouped.groups.keys())
        elif isinstance(sort_by, list):
            # Custom order provided as list
            available_stations = set(grouped.groups.keys())
            station_order = [s for s in sort_by if s in available_stations]
            # Add any missing stations at the end
            missing_stations = sorted(available_stations - set(station_order))
            station_order.extend(missing_stations)
        elif sort_by in ['total_picks', 'total_p', 'total_s', 'best', 'best_p', 'best_s']:
            # Sort by pick counts or timing precision
            station_metrics = []
            for station, station_group in grouped:
                p_data = station_group[station_group['phase_hint'] == 'P']['delta_t']
                s_data = station_group[station_group['phase_hint'] == 'S']['delta_t']
                
                p_count = len(p_data)
                s_count = len(s_data)
                total_count = len(station_group)
                
                p_std = p_data.std() if p_count > 0 else float('inf')
                s_std = s_data.std() if s_count > 0 else float('inf')
                
                if sort_by == 'total_picks':
                    metric = total_count
                elif sort_by == 'total_p':
                    metric = p_count
                elif sort_by == 'total_s':
                    metric = s_count
                elif sort_by == 'best':
                    # Combined metric: average of P and S std (lower is better)
                    valid_stds = [std for std in [p_std, s_std] if std != float('inf')]
                    metric = sum(valid_stds) / len(valid_stds) if valid_stds else float('inf')
                elif sort_by == 'best_p':
                    metric = p_std
                elif sort_by == 'best_s':
                    metric = s_std
                    
                station_metrics.append((station, metric))
            
            # Sort by metric (ascending for counts, descending for std) then by station name for ties
            if sort_by in ['total_picks', 'total_p', 'total_s']:
                # Lower counts first (ascending)
                station_metrics.sort(key=lambda x: (x[1], x[0]))
            else:
                # Higher std first (descending)
                station_metrics.sort(key=lambda x: (-x[1], x[0]))
            station_order = [station for station, _ in station_metrics]
        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}. Must be None, list, or one of: 'total_picks', 'total_p', 'total_s', 'best', 'best_p', 'best_s'")
        
        n_stations = len(station_order)
        n_cols = 4
        n_rows = (n_stations + n_cols - 1) // n_cols

        # Calculate figure height if not provided
        if figsize[1] is None:
            figsize = (figsize[0], 4 * n_rows)

        # Find global min/max for x-axis
        global_max = abs(df['delta_t']).max()
        x_range = (-global_max, global_max)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        plt.suptitle('Pick Time Differences by Station\nNegative values: catalog 1 arrival is early')

        for i, station in enumerate(station_order):
            station_group = grouped.get_group(station)
            
            # Plot P phase in red (if included)
            if 'P' in include_phases:
                p_data = station_group[station_group['phase_hint'] == 'P']['delta_t']
                p_count = len(p_data)
                p_std = p_data.std() if p_count > 0 else 0
                
                if p_count > 0:
                    axes[i].hist(p_data, bins=np.arange(x_range[0], x_range[1] + 0.25, 0.25),
                                 alpha=0.6, color='red', label=f'P (n={p_count}, σ={p_std:.2f}s)')

            # Plot S phase in blue (if included)
            if 'S' in include_phases:
                s_data = station_group[station_group['phase_hint'] == 'S']['delta_t']
                s_count = len(s_data)
                s_std = s_data.std() if s_count > 0 else 0
                
                if s_count > 0:
                    axes[i].hist(s_data, bins=np.arange(x_range[0], x_range[1] + 0.25, 0.25),
                                 alpha=0.6, color='blue', label=f'S (n={s_count}, σ={s_std:.2f}s)')

            axes[i].set_title(f'Station: {station}')
            axes[i].set_xlabel('Δt (seconds)')
            axes[i].set_ylabel('Count')
            axes[i].grid(True)
            axes[i].set_xlim(x_range)
            
            # Position legend outside the axes at the bottom, horizontally
            axes[i].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the legends
        
        return fig, axes

    @staticmethod
    def plot_pick_deltas_boxplot(df, figsize=(10, None), sort_by=None, include_phases=['P', 'S'], ax=None):
        """
        Plot box and whisker plots of pick time deltas for various stations.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame from get_pick_deltas() containing
            columns: station, phase_hint, delta_t
        figsize : tuple, optional
            Figure size as (width, height). If height is None, it will be 
            calculated based on number of stations (default: (10, None))
        sort_by : str, list, or None, optional
            How to sort stations. Options:
            - None: alphabetical order (default)
            - 'total_picks': sort by total number of picks (ascending)
            - 'total_p': sort by number of P picks (ascending) 
            - 'total_s': sort by number of S picks (ascending)
            - 'best': sort by overall timing precision (highest combined std first, descending)
            - 'best_p': sort by P timing precision (highest P std first, descending)
            - 'best_s': sort by S timing precision (highest S std first, descending)
            - list: custom order of station codes
        include_phases : list, optional
            List of phase types to include in the plot (default: ['P', 'S'])
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure and axes (default: None)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        """
        # Group data by station and determine station order
        grouped = df.groupby('station')
        
        # Determine station ordering (same logic as histogram function)
        if sort_by is None:
            # Default: alphabetical order
            station_order = sorted(grouped.groups.keys())
        elif isinstance(sort_by, list):
            # Custom order provided as list
            available_stations = set(grouped.groups.keys())
            station_order = [s for s in sort_by if s in available_stations]
            # Add any missing stations at the end
            missing_stations = sorted(available_stations - set(station_order))
            station_order.extend(missing_stations)
        elif sort_by in ['total_picks', 'total_p', 'total_s', 'best', 'best_p', 'best_s']:
            # Sort by pick counts or timing precision
            station_metrics = []
            for station, station_group in grouped:
                p_data = station_group[station_group['phase_hint'] == 'P']['delta_t']
                s_data = station_group[station_group['phase_hint'] == 'S']['delta_t']
                
                p_count = len(p_data)
                s_count = len(s_data)
                total_count = len(station_group)
                
                p_std = p_data.std() if p_count > 0 else float('inf')
                s_std = s_data.std() if s_count > 0 else float('inf')
                
                if sort_by == 'total_picks':
                    metric = total_count
                elif sort_by == 'total_p':
                    metric = p_count
                elif sort_by == 'total_s':
                    metric = s_count
                elif sort_by == 'best':
                    # Combined metric: average of P and S std (lower is better)
                    valid_stds = [std for std in [p_std, s_std] if std != float('inf')]
                    metric = sum(valid_stds) / len(valid_stds) if valid_stds else float('inf')
                elif sort_by == 'best_p':
                    metric = p_std
                elif sort_by == 'best_s':
                    metric = s_std
                    
                station_metrics.append((station, metric))
            
            # Sort by metric (ascending for counts, descending for std) then by station name for ties
            if sort_by in ['total_picks', 'total_p', 'total_s']:
                # Lower counts first (ascending)
                station_metrics.sort(key=lambda x: (x[1], x[0]))
            else:
                # Higher std first (descending)
                station_metrics.sort(key=lambda x: (-x[1], x[0]))
            station_order = [station for station, _ in station_metrics]
        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}. Must be None, list, or one of: 'total_picks', 'total_p', 'total_s', 'best', 'best_p', 'best_s'")
        
        # Collect data for box plots in the correct order
        stations = []
        p_data_list = []
        s_data_list = []
        p_counts = []
        s_counts = []

        for station in station_order:
            station_group = grouped.get_group(station)
            stations.append(station)
            
            # P phase data
            p_data = station_group[station_group['phase_hint'] == 'P']['delta_t']
            s_data = station_group[station_group['phase_hint'] == 'S']['delta_t']
            
            p_data_list.append(p_data.values if len(p_data) > 0 else [])
            s_data_list.append(s_data.values if len(s_data) > 0 else [])
            p_counts.append(len(p_data))
            s_counts.append(len(s_data))

        # Handle axes parameter
        created_new_figure = False
        if ax is None:
            # Calculate figure height if not provided
            if figsize[1] is None:
                figsize = (figsize[0], len(stations) * 0.8)
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize)
            created_new_figure = True
        else:
            # Use provided axes
            fig = ax.figure

        # Set up y positions for box plots
        y_pos = np.arange(len(stations))
        box_width = 0.3

        # Create box plots for each station
        for i, (station, p_data, s_data, p_count, s_count) in enumerate(zip(stations, p_data_list, s_data_list, p_counts, s_counts)):
            
            # P phase box plot (red) - positioned above (if included)
            if 'P' in include_phases and len(p_data) > 0:
                bp_p = ax.boxplot(p_data, positions=[i + box_width/2], widths=box_width/2, 
                                 vert=False, patch_artist=True,
                                 boxprops=dict(facecolor='red', alpha=0.7),
                                 medianprops=dict(color='darkred', linewidth=2))
                
                # Add count label - position relative to axes (responsive to xlim changes)
                ax.text(0.95, i + box_width/2, f'N = {p_count}', 
                       va='center', ha='right', fontsize=10, color='red',
                       transform=ax.get_yaxis_transform())
            
            # S phase box plot (blue) - positioned below (if included)
            if 'S' in include_phases and len(s_data) > 0:
                bp_s = ax.boxplot(s_data, positions=[i - box_width/2], widths=box_width/2, 
                                 vert=False, patch_artist=True,
                                 boxprops=dict(facecolor='blue', alpha=0.7),
                                 medianprops=dict(color='darkblue', linewidth=2))
                
                # Add count label - position relative to axes (responsive to xlim changes)
                ax.text(0.95, i - box_width/2, f'N = {s_count}', 
                       va='center', ha='right', fontsize=10, color='blue',
                       transform=ax.get_yaxis_transform())

        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stations)
        ax.set_xlabel('Δt (seconds)')
        ax.set_title('Distribution of Pick Time Differences by Station')

        # Set x-axis to be centered on 0
        all_data = []
        for p_data, s_data in zip(p_data_list, s_data_list):
            if len(p_data) > 0:
                all_data.extend(p_data)
            if len(s_data) > 0:
                all_data.extend(s_data)

        if all_data:
            data_range = max(abs(min(all_data)), abs(max(all_data)))
            ax.set_xlim(-data_range * 1.2, data_range * 1.2)

        # Set y-axis limits to ensure bottom spine is below all box plots
        ax.set_ylim(-0.5, len(stations) - 0.5)

        # Remove spines except bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_position(('data', -0.5))  # Position at bottom of plot area

        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linewidth=1)

        # Create custom legend (only for included phases)
        from matplotlib.patches import Patch
        legend_elements = []
        if 'P' in include_phases:
            legend_elements.append(Patch(facecolor='red', alpha=0.7, label='P'))
        if 'S' in include_phases:
            legend_elements.append(Patch(facecolor='blue', alpha=0.7, label='S'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right')

        # Only apply tight_layout if we created a new figure
        if created_new_figure:
            plt.tight_layout()
        
        return fig, ax
