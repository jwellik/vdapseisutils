import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.collections import PathCollection
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
from scipy.signal import welch
from obspy import Stream, Trace, UTCDateTime
from obspy.imaging.util import _set_xaxis_obspy_dates
import matplotlib.patches as patches
import warnings

# TODO thresholds are always added even if provided as None


def _tight_layout_safe(fig):
    """
    Call tight_layout on a figure, suppressing the UserWarning about
    incompatible axes (e.g., from GridSpec layouts).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                               message='.*tight_layout.*')
        fig.tight_layout()


class Buurman_2010:
    """
    Recreates Buurman and West (2010) figures showing spectral analysis and frequency index.
    https://pubs.usgs.gov/publication/pp1769, Chapter 2, Seismic Precursors to Volcanic Explosions During the
2006 Eruption of Augustine Volcano
    
    This class provides methods to create different figure layouts:
    - plot_fig2(): Waveform axes only (Figure 2 from Buurman et al.)
    - plot_fig3(): Spectra and frequency index axes (Figure 3 from Buurman et al.)
    - plot_fig23(): Combined layout with waveform, spectra, and frequency index
    - plot_fig4(): Timeseries of frequency index over time
    - plot_fig234(): Combined layout with waveform, spectra, frequency index, and timeseries
    - plot_fi_magnitude(): FI vs Magnitude scatter plot (requires magnitude values)

    Static Methods (can be used without creating an instance):
    - calculate_spectrum(): Calculate amplitude spectrum from a trace
    - calculate_frequency_index(): Calculate frequency index from a trace
    - extract_catalog_data(): Extract times, FI values, and magnitudes from catalog
    - plot_fi_timeseries_static(): Plot FI timeseries from pre-computed data
    - plot_fi_magnitude_static(): Plot FI vs Magnitude from pre-computed data
    
    Class Methods:
    - from_catalog(): Create instance and populate from ObsPy Catalog

    Parameters:
    -----------
    Alower : list of float
        Lower frequency bounds for spectral regions [Hz]
    Aupper : list of float
        Upper frequency bounds for spectral regions [Hz]
    thresholds : list of float or None
        Thresholds for FI-based earthquake classification. These default values
        ([-1.3, -0.4]) are from the original Buurman and West (2010) publication.
        Events are classified as:
        - Low frequency: frequency index < lowest threshold
        - Hybrid: frequency index between thresholds
        - High frequency: frequency index > highest threshold
        If None, no automatic color coding will be applied (uses black).
    threshold_colors : list of str or None
        Colors for each classification region. Must have one more element than
        thresholds (one color for each region plus one for each threshold).
        Default: ["black", "red", "blue"] for low frequency, hybrid, high frequency.
        If None or if any element is None, black will be used for that region.
    analysis_window : list of float or None
        Time window in seconds [start, end] to use for spectrum and frequency
        index calculation. If None, uses the entire waveform (default: None).
        Example: [1, 5] uses data from 1 to 5 seconds after waveform start.
    highlight_color : str
        Color for highlighted regions (frequency bands in spectrum, analysis_window in waveform).
        Default: "lightgreen"
    highlight_alpha : float
        Alpha (transparency) for highlighted regions. Default: 0.3
    
    Notes:
    ------
    To create FI vs Magnitude plots, provide magnitude values when adding streams
    using add_stream(magnitude=...) or add_streams_with_magnitudes(). Magnitudes
    should be provided in the same order as the streams.
    
    For plot_fig234(), you can provide an example_waveform for panels A-C and
    separate timeseries data (from catalog or pre-computed) for panel D. This allows
    you to show a stacked or example waveform while displaying individual event
    data in the timeseries panel.
    """

    def __init__(self, Alower=None, Aupper=None, thresholds=None, threshold_colors=None,
                 analysis_window=None, highlight_color="lightgreen",
                 highlight_alpha=0.3):
        # Store parameters
        self.Alower = Alower or [1, 2]
        self.Aupper = Aupper or [10, 20]
        # Allow thresholds to be None explicitly
        if thresholds is None:
            self.thresholds = [-1.3, -0.4]  # Default from original publication
        else:
            self.thresholds = thresholds
        self.analysis_window = analysis_window
        self.highlight_color = highlight_color
        self.highlight_alpha = highlight_alpha

        # Ensure threshold_colors has one more element than thresholds
        if threshold_colors is None:
            if self.thresholds is not None:
                self.threshold_colors = ["black", "red", "blue"]  # Default: 3 colors for 2 thresholds
            else:
                self.threshold_colors = None
        else:
            if self.thresholds is not None and len(threshold_colors) != len(self.thresholds) + 1:
                raise ValueError(f"threshold_colors must have {len(self.thresholds) + 1} elements "
                                 f"(one more than thresholds), got {len(threshold_colors)}")
            self.threshold_colors = threshold_colors

        # Initialize figure and axes (will be set by plot methods)
        self.fig = None
        self.ax_wave = None
        self.ax1 = None  # Spectrum axis
        self.ax2 = None  # Frequency index axis
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        # Store streams for later plotting (for waveform/spectrum/FI panels)
        self._streams = []  # List of dicts: {'trace': Trace, 'color': str, 'analysis_window': list or None, 'magnitude': float or None}
        
        # Store catalogs for timeseries and FI vs Magnitude panels
        self._catalogs = []  # List of dicts: {'times': list, 'fi_values': list, 'magnitudes': list or None, 'scatter_kwargs': dict}
        
        # Default scatter parameters for catalogs
        # Use facecolor instead of color to ensure edgecolor works properly
        self._default_scatter_kwargs = {
            'marker': 'o',
            'facecolor': 'k',
            'edgecolor': 'k',
            'linewidth': 0.5,
            'alpha': 0.3,
            's': 30
        }
        
        # Store custom axis limits (None means use defaults)
        self._wave_xlim = None
        self._spectra_xlim = None
        self._spectra_ylim = None
        self._fi_ylim = None
        
        # Waveform spacing mode for fig2 (set by plot_fig2)
        self._waveform_spacing_mode = "center"
        self._waveform_spacing_center = 0.0

    @staticmethod
    def calculate_spectrum(trace, analysis_window=None):
        """
        Calculate amplitude spectrum from a trace.
        
        Uses Welch's method to compute power spectral density, then converts to
        amplitude spectrum (matching FFT-based approach used by Buurman & West 2010).
        
        This is a static method that can be used independently of an instance.
        
        Parameters:
        -----------
        trace : obspy.Trace
            ObsPy Trace object containing waveform data.
        analysis_window : list of float or None, optional
            Time window in seconds [start, end] to use for spectrum calculation.
            If None, uses the entire waveform.
            
        Returns:
        --------
        freqs : numpy.ndarray
            Frequency array [Hz]
        amplitude_normalized : numpy.ndarray
            Normalized amplitude spectrum (area under curve = 1)
        """
        tr = trace
        
        # Prepare data for spectrum calculation
        if analysis_window is not None:
            if len(analysis_window) != 2:
                raise ValueError("analysis_window must be a 2-element list [start, end]")
            start_time, end_time = analysis_window
            sampling_rate = tr.stats.sampling_rate
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            end_sample = min(end_sample, len(tr.data))
            tr_for_analysis = tr.copy()
            tr_for_analysis.data = tr.data[start_sample:end_sample]
        else:
            tr_for_analysis = tr
        
        # Calculate power spectral density using Welch's method
        # Use full data length to match FFT resolution (as in Buurman & West 2010)
        sampling_rate = tr_for_analysis.stats.sampling_rate
        nperseg = len(tr_for_analysis.data)
        
        freqs, psd = welch(tr_for_analysis.data, fs=sampling_rate, nperseg=nperseg,
                           scaling='density', detrend='linear')
        
        # Remove DC component
        freqs = freqs[1:]
        psd = psd[1:]
        
        # Convert PSD (power) to amplitude spectrum by taking square root
        # This matches the FFT-based approach used by Buurman & West (2010)
        amplitude = np.sqrt(psd)
        
        # Normalize by area under curve
        area = np.trapezoid(amplitude, freqs)
        amplitude_normalized = amplitude / area
        
        return freqs, amplitude_normalized

    @staticmethod
    def calculate_frequency_index(trace, Alower, Aupper, analysis_window=None):
        """
        Calculate frequency index from a trace.
        
        This is a static method that can be used independently of an instance.
        
        Parameters:
        -----------
        trace : obspy.Trace
            ObsPy Trace object containing waveform data.
        Alower : list of float
            Lower frequency bounds for spectral regions [Hz]
        Aupper : list of float
            Upper frequency bounds for spectral regions [Hz]
        analysis_window : list of float or None, optional
            Time window in seconds [start, end] to use for frequency index calculation.
            If None, uses the entire waveform.
            
        Returns:
        --------
        float
            Frequency index value
        """
        # Calculate spectrum
        freqs, amplitude_normalized = Buurman_2010.calculate_spectrum(trace, analysis_window)
        
        # Calculate frequency index
        lower_mask = (freqs >= Alower[0]) & (freqs < Alower[1])
        upper_mask = (freqs >= Aupper[0]) & (freqs < Aupper[1])
        
        freq_index = np.log10(np.mean(amplitude_normalized[upper_mask]) / np.mean(amplitude_normalized[lower_mask]))
        
        return freq_index

    @staticmethod
    def extract_catalog_data(catalog, event_traces, Alower, Aupper, analysis_window=None):
        """
        Extract times, FI values, and magnitudes from catalog and event traces.
        
        Parameters:
        -----------
        catalog : obspy.Catalog
            ObsPy Catalog object containing event metadata.
        event_traces : list of obspy.Trace
            List of traces corresponding to events in catalog. Must match catalog length.
        Alower : list of float
            Lower frequency bounds for spectral regions [Hz]
        Aupper : list of float
            Upper frequency bounds for spectral regions [Hz]
        analysis_window : list of float or None, optional
            Time window in seconds [start, end] to use for frequency index calculation.
            
        Returns:
        --------
        times : list of obspy.UTCDateTime
            Event origin times
        fi_values : list of float
            Frequency index values for each event
        magnitudes : list of float or None
            Magnitude values for each event (None if not available)
        """
        if len(catalog) != len(event_traces):
            raise ValueError(f"Catalog has {len(catalog)} events but {len(event_traces)} traces provided")
        
        times = []
        fi_values = []
        magnitudes = []
        
        for event, trace in zip(catalog, event_traces):
            # Get origin time
            if event.origins and len(event.origins) > 0:
                times.append(event.origins[0].time)
            else:
                # Fallback to trace starttime
                times.append(trace.stats.starttime)
            
            # Calculate frequency index
            fi = Buurman_2010.calculate_frequency_index(trace, Alower, Aupper, analysis_window)
            fi_values.append(fi)
            
            # Get magnitude (prefer preferred magnitude, otherwise first available)
            mag = None
            if event.magnitudes and len(event.magnitudes) > 0:
                # Try to find preferred magnitude
                for m in event.magnitudes:
                    if m.magnitude_type in ['ML', 'Mw', 'Ms', 'Mb']:
                        mag = m.mag
                        break
                # If no preferred found, use first
                if mag is None:
                    mag = event.magnitudes[0].mag
            magnitudes.append(mag)
        
        return times, fi_values, magnitudes

    @classmethod
    def from_catalog(cls, catalog, event_traces, Alower=None, Aupper=None, thresholds=None,
                     threshold_colors=None, analysis_window=None, highlight_color="lightgreen",
                     highlight_alpha=0.3):
        """
        Create instance and populate from catalog.
        
        This class method creates a Buurman_2010 instance and automatically extracts
        event data from a catalog and corresponding traces.
        
        Parameters:
        -----------
        catalog : obspy.Catalog
            ObsPy Catalog object containing event metadata.
        event_traces : list of obspy.Trace
            List of traces corresponding to events in catalog. Must match catalog length.
        Alower : list of float, optional
            Lower frequency bounds for spectral regions [Hz]. Default: [1, 2]
        Aupper : list of float, optional
            Upper frequency bounds for spectral regions [Hz]. Default: [10, 20]
        thresholds : list of float or None, optional
            Thresholds for FI-based classification. Default: [-1.3, -0.4]
        threshold_colors : list of str or None, optional
            Colors for each classification region.
        analysis_window : list of float or None, optional
            Time window in seconds [start, end] to use for spectrum and frequency
            index calculation.
        highlight_color : str, optional
            Color for highlighted regions. Default: "lightgreen"
        highlight_alpha : float, optional
            Alpha (transparency) for highlighted regions. Default: 0.3
            
        Returns:
        --------
        Buurman_2010
            Instance with streams populated from catalog
        """
        # Create instance
        instance = cls(Alower=Alower, Aupper=Aupper, thresholds=thresholds,
                      threshold_colors=threshold_colors, analysis_window=analysis_window,
                      highlight_color=highlight_color, highlight_alpha=highlight_alpha)
        
        # Extract data from catalog
        times, fi_values, magnitudes = cls.extract_catalog_data(
            catalog, event_traces, instance.Alower, instance.Aupper, analysis_window)
        
        # Add streams to instance
        for trace, mag in zip(event_traces, magnitudes):
            instance.add_stream(trace, magnitude=mag, analysis_window=analysis_window)
        
        return instance

    def plot_fig2(self, figsize=(8, 4), waveform_spacing="center"):
        """
        Create Figure 2 layout: waveform axes only.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 4)
            Figure size (width, height) in inches.
        waveform_spacing : str or float, default="center"
            Controls how waveforms are spaced along the y-axis:
            - "fi": Space waveforms by their frequency index values
            - "even": Space waveforms evenly along the y-axis
            - "center" or float: All waveforms centered at the same y-value.
              If "center", uses default value 0. If a float is provided,
              uses that value as the center.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        # Reset state
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        # Set waveform spacing mode
        if isinstance(waveform_spacing, (int, float)):
            self._waveform_spacing_mode = "center"
            self._waveform_spacing_center = float(waveform_spacing)
        elif waveform_spacing.lower() == "fi":
            self._waveform_spacing_mode = "fi"
            self._waveform_spacing_center = 0.0
        elif waveform_spacing.lower() == "even":
            self._waveform_spacing_mode = "even"
            self._waveform_spacing_center = 0.0
        elif waveform_spacing.lower() == "center":
            self._waveform_spacing_mode = "center"
            self._waveform_spacing_center = 0.0
        else:
            raise ValueError(f"waveform_spacing must be 'fi', 'even', 'center', or a float, got {waveform_spacing}")
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        
        # Single axis for waveform
        self.ax_wave = self.fig.add_subplot(1, 1, 1)
        self.ax_wave.set_ylabel('AMPLITUDE', fontsize=10, color='grey')
        self.ax_wave.set_xlabel('TIME, IN SECONDS', fontsize=10, color='grey')
        self.ax_wave.set_yticks([])
        self.ax_wave.tick_params(axis='both', colors='grey', labelsize=10)
        
        # Add panel label
        self.ax_wave.text(0.02, 0.98, 'A.', transform=self.ax_wave.transAxes,
                         fontsize=10, color='black', weight='bold',
                         verticalalignment='top', horizontalalignment='left')
        
        # No spectrum or frequency index axes for fig2
        self.ax1 = None
        self.ax2 = None
        
        # For "even" spacing, we need to collect all waveforms first to calculate positions
        if self._waveform_spacing_mode == "even":
            # First pass: calculate FI for all streams to determine spacing
            fi_values = []
            for stream_info in self._streams:
                fi = self._calculate_fi_for_stream(stream_info)
                fi_values.append(fi)
            
            # Calculate evenly spaced positions
            n_waveforms = len(self._streams)
            if n_waveforms > 1:
                # Space evenly with some padding
                spacing = 2.0  # Distance between waveform centers
                total_range = (n_waveforms - 1) * spacing
                start_y = -total_range / 2
                self._waveform_y_positions = [start_y + i * spacing for i in range(n_waveforms)]
            else:
                self._waveform_y_positions = [0.0]
        else:
            self._waveform_y_positions = None
        
        _tight_layout_safe(self.fig)
        
        # Plot all stored streams
        for i, stream_info in enumerate(self._streams):
            if self._waveform_spacing_mode == "even":
                # Pass the y-position for this waveform
                self._plot_stream(stream_info, waveform_y_position=self._waveform_y_positions[i])
            else:
                self._plot_stream(stream_info)
        
        # Set y-axis limits based on spacing mode (after all waveforms are plotted)
        if self._waveform_spacing_mode == "fi" and len(self._waveform_freq_indices) > 0:
            # Use FI-based limits
            min_fi = min(self._waveform_freq_indices)
            max_fi = max(self._waveform_freq_indices)
            y_padding = 1.5
            self.ax_wave.set_ylim(min_fi - y_padding, max_fi + y_padding)
        elif self._waveform_spacing_mode == "even" and self._waveform_y_positions is not None:
            # Use evenly spaced positions
            min_y = min(self._waveform_y_positions) - 1.5
            max_y = max(self._waveform_y_positions) + 1.5
            self.ax_wave.set_ylim(min_y, max_y)
        elif self._waveform_spacing_mode == "center":
            # Center mode - set limits around center with padding
            y_padding = 1.5
            self.ax_wave.set_ylim(self._waveform_spacing_center - y_padding,
                                 self._waveform_spacing_center + y_padding)
        
        # Apply custom limits if set
        self._apply_limits()
        
        plt.draw()
        return self.fig

    def plot_fig3(self, figsize=(8, 6)):
        """
        Create Figure 3 layout: spectra and frequency index axes (no waveform).
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 6)
            Figure size (width, height) in inches.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        # Reset state
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        
        # GridSpec for spectrum and frequency index
        gs = gridspec.GridSpec(1, 2, figure=self.fig,
                               width_ratios=[7, 1], wspace=0.2)
        
        # Left: Spectrum
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        
        # Right: Frequency Index
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        
        # No waveform axis for fig3
        self.ax_wave = None
        
        # Setup spectrum subplot
        self.ax1.set_xlabel('FREQUENCY (HZ)', fontsize=10, color='grey')
        self.ax1.set_ylabel('AMPLITUDE, NORMALIZED', fontsize=10, color='grey')
        self.ax1.set_xscale('log')
        self.ax1.set_xlim(0.1, 50)
        self.ax1.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label A.
        self.ax1.text(0.02, 0.98, 'A.', transform=self.ax1.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')

        # Add shaded regions for frequency bands (using 'k' to match figure_for_margarita)
        if len(self.Alower) >= 1 and len(self.Aupper) >= 1:
            self.ax1.axvspan(self.Alower[0], self.Alower[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{lower}$')
        if len(self.Alower) >= 2 and len(self.Aupper) >= 2:
            self.ax1.axvspan(self.Aupper[0], self.Aupper[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{upper}$')

        # Setup frequency index subplot
        self.ax2.set_xlabel('FI', fontsize=10, color='grey')
        self.ax2.set_ylabel('')
        self.ax2.set_xlim(-0.5, 0.5)
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')
        self.ax2.set_xticks([])
        self.ax2.set_xticklabels([])
        self.ax2.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label B.
        self.ax2.text(0.02, 0.98, 'B.', transform=self.ax2.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')

        # Add threshold lines (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax2.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)

        _tight_layout_safe(self.fig)
        
        # Plot all stored streams
        for stream_info in self._streams:
            self._plot_stream(stream_info)
        
        # Apply custom limits if set
        self._apply_limits()
        
        plt.draw()
        return self.fig

    def plot_fig23(self, figsize=(8, 8)):
        """
        Create combined Figure 2+3 layout: waveform, spectra, and frequency index axes.
        This mimics the original default behavior.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 8)
            Figure size (width, height) in inches.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        # Reset state
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        
        # Use GridSpec with waveform on top (waveform is half height of spectra)
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[2.5, 5], width_ratios=[7, 1],
                               hspace=0.3, wspace=0.2)
        
        # Top: Waveform (spans both columns)
        self.ax_wave = self.fig.add_subplot(gs[0, :])
        self.ax_wave.set_ylabel('AMPLITUDE', fontsize=10, color='grey')
        self.ax_wave.set_xlabel('')
        self.ax_wave.set_yticks([])
        self.ax_wave.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label A.
        self.ax_wave.text(0.02, 0.98, 'A.', transform=self.ax_wave.transAxes,
                         fontsize=10, color='black', weight='bold',
                         verticalalignment='top', horizontalalignment='left')
        
        # Bottom left: Spectrum
        self.ax1 = self.fig.add_subplot(gs[1, 0])
        
        # Bottom right: Frequency Index
        self.ax2 = self.fig.add_subplot(gs[1, 1])

        # Setup spectrum subplot
        self.ax1.set_xlabel('FREQUENCY (HZ)', fontsize=10, color='grey')
        self.ax1.set_ylabel('AMPLITUDE, NORMALIZED', fontsize=10, color='grey')
        self.ax1.set_xscale('log')
        self.ax1.set_xlim(0.1, 50)
        self.ax1.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label B.
        self.ax1.text(0.02, 0.98, 'B.', transform=self.ax1.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')

        # Add shaded regions for frequency bands (using 'k' to match figure_for_margarita)
        if len(self.Alower) >= 1 and len(self.Aupper) >= 1:
            self.ax1.axvspan(self.Alower[0], self.Alower[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{lower}$')
        if len(self.Alower) >= 2 and len(self.Aupper) >= 2:
            self.ax1.axvspan(self.Aupper[0], self.Aupper[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{upper}$')

        # Setup frequency index subplot
        self.ax2.set_xlabel('FI', fontsize=10, color='grey')
        self.ax2.set_ylabel('')
        self.ax2.set_xlim(-0.5, 0.5)
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')
        self.ax2.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label C.
        self.ax2.text(0.02, 0.98, 'C.', transform=self.ax2.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')

        # Add threshold lines (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax2.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)

        _tight_layout_safe(self.fig)
        
        # Plot all stored streams
        for stream_info in self._streams:
            self._plot_stream(stream_info)
        
        # Apply custom limits if set
        self._apply_limits()
        
        plt.draw()
        return self.fig

    def plot_fig4(self, figsize=(8, 6)):
        """
        Create Figure 4 layout: timeseries of FI over time.
        
        This method plots frequency index values as a function of time for all
        stored streams. Each point is plotted as a scatter point with the same
        parameters as used in figure3. Threshold lines are drawn as horizontal
        axes. The x-axis uses ObsPy UTCDateTime objects and is formatted using
        ObsPy's date formatting function.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 6)
            Figure size (width, height) in inches.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        
        # Single axis for FI timeseries
        self.ax_fi_timeseries = self.fig.add_subplot(1, 1, 1)
        
        # Setup axis labels
        self.ax_fi_timeseries.set_xlabel('TIME', fontsize=10, color='grey')
        self.ax_fi_timeseries.set_ylabel('FI', fontsize=10, color='grey')
        self.ax_fi_timeseries.tick_params(axis='both', colors='grey', labelsize=10)
        
        # Add threshold lines (if thresholds are set) - same as figure3
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax_fi_timeseries.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        # Collect FI values and times for all streams
        times = []
        fi_values = []
        colors = []
        
        for stream_info in self._streams:
            tr = stream_info['trace']
            color = stream_info['color']
            analysis_window_override = stream_info['analysis_window']
            
            # Determine which analysis_window to use
            window_to_use = analysis_window_override if analysis_window_override is not None else self.analysis_window
            
            # Prepare data for spectrum calculation (same logic as _plot_stream)
            if window_to_use is not None:
                if len(window_to_use) != 2:
                    raise ValueError("analysis_window must be a 2-element list [start, end]")
                start_time, end_time = window_to_use
                sampling_rate = tr.stats.sampling_rate
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                end_sample = min(end_sample, len(tr.data))
                tr_for_analysis = tr.copy()
                tr_for_analysis.data = tr.data[start_sample:end_sample]
            else:
                tr_for_analysis = tr
            
            # Calculate power spectral density using Welch's method
            # Use full data length to match FFT resolution (as in Buurman & West 2010)
            sampling_rate = tr_for_analysis.stats.sampling_rate
            nperseg = len(tr_for_analysis.data)
            
            freqs, psd = welch(tr_for_analysis.data, fs=sampling_rate, nperseg=nperseg,
                               scaling='density', detrend='linear')
            
            # Remove DC component
            freqs = freqs[1:]
            psd = psd[1:]
            
            # Convert PSD (power) to amplitude spectrum by taking square root
            # This matches the FFT-based approach used by Buurman & West (2010)
            amplitude = np.sqrt(psd)
            
            # Normalize by area under curve
            area = np.trapezoid(amplitude, freqs)
            amplitude_normalized = amplitude / area
            
            # Calculate frequency index
            freq_index = self._calculate_frequency_index(freqs, amplitude_normalized)
            
            # Get time (use starttime of the trace)
            time = tr.stats.starttime
            
            # Determine color based on user preference
            if color == "auto":
                plot_color = self._get_color_from_freq_index(freq_index)
            else:
                plot_color = color
            
            times.append(time)
            fi_values.append(freq_index)
            colors.append(plot_color)
        
        # Plot as scatter plot (using 'k' for all points to match figure_for_margarita)
        if len(times) > 0:
            # Convert times to matplotlib dates for plotting
            time_dates = [t.matplotlib_date for t in times]
            self.ax_fi_timeseries.scatter(time_dates, fi_values, c='k', alpha=0.7, s=30, 
                                         edgecolor='k', linewidth=0.5, marker='o')
            
            # Format x-axis using ObsPy's date formatting
            _set_xaxis_obspy_dates(self.ax_fi_timeseries)
        
        _tight_layout_safe(self.fig)
        plt.draw()
        return self.fig

    def plot_fig234(self, figsize=(8, 10), example_waveform=None, catalog=None,
                    event_traces=None, times=None, fi_values=None, magnitudes=None):
        """
        Create combined Figure 2+3+4 layout: waveform, spectra, frequency index, and timeseries.
        
        This method creates a layout with:
        - Top: Waveform axes (spans full width) - shows example_waveform if provided
        - Middle: Spectra (left) and frequency index (right) axes - shows example_waveform if provided
        - Bottom: Timeseries of FI over time (spans full width) - shows individual events
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 10)
            Figure size (width, height) in inches.
        example_waveform : obspy.Trace or None, optional
            Trace to display in waveform, spectra, and FI panels (panels A-C).
            If None, uses stored streams (backward compatibility).
        catalog : obspy.Catalog or None, optional
            Catalog object for extracting event metadata. Requires event_traces.
            Used for timeseries panel (panel D).
        event_traces : list of obspy.Trace or None, optional
            Traces corresponding to catalog events. Required if catalog is provided.
            Used for timeseries panel (panel D).
        times : list or array-like or None, optional
            Pre-computed time values for timeseries panel. Can be UTCDateTime objects,
            matplotlib dates, or datetime objects.
        fi_values : list or array-like or None, optional
            Pre-computed frequency index values for timeseries panel.
        magnitudes : list or array-like or None, optional
            Pre-computed magnitude values for timeseries panel.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
            
        Notes:
        ------
        For the timeseries panel (panel D), you can provide data in one of three ways:
        1. Provide catalog + event_traces (will compute FI from traces)
        2. Provide times + fi_values + magnitudes (pre-computed data)
        3. Use stored streams (backward compatibility, default behavior)
        """
        # Reset state
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        
        # Use GridSpec with 3 rows: waveform, spectra/FI, timeseries
        # Height ratios: waveform (2.5), spectra/FI (5), timeseries (3)
        gs = gridspec.GridSpec(3, 2, figure=self.fig,
                               height_ratios=[2.5, 5, 3], width_ratios=[7, 1],
                               hspace=0.3, wspace=0.2)
        
        # Top row: Waveform (spans both columns)
        self.ax_wave = self.fig.add_subplot(gs[0, :])
        self.ax_wave.set_ylabel('AMPLITUDE', fontsize=10, color='grey')
        self.ax_wave.set_xlabel('')
        self.ax_wave.set_yticks([])
        self.ax_wave.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label A.
        self.ax_wave.text(0.02, 0.98, 'A.', transform=self.ax_wave.transAxes,
                         fontsize=10, color='black', weight='bold',
                         verticalalignment='top', horizontalalignment='left')
        
        # Middle row left: Spectrum
        self.ax1 = self.fig.add_subplot(gs[1, 0])
        
        # Middle row right: Frequency Index
        self.ax2 = self.fig.add_subplot(gs[1, 1])
        
        # Bottom row: Timeseries (spans both columns)
        self.ax_fi_timeseries = self.fig.add_subplot(gs[2, :])
        
        # Setup spectrum subplot
        self.ax1.set_xlabel('FREQUENCY (HZ)', fontsize=10, color='grey')
        self.ax1.set_ylabel('AMPLITUDE, NORMALIZED', fontsize=10, color='grey')
        self.ax1.set_xscale('log')
        self.ax1.set_xlim(0.1, 50)
        self.ax1.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label B.
        self.ax1.text(0.02, 0.98, 'B.', transform=self.ax1.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')
        
        # Add shaded regions for frequency bands (using 'k' to match figure_for_margarita)
        if len(self.Alower) >= 1 and len(self.Aupper) >= 1:
            self.ax1.axvspan(self.Alower[0], self.Alower[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{lower}$')
        if len(self.Alower) >= 2 and len(self.Aupper) >= 2:
            self.ax1.axvspan(self.Aupper[0], self.Aupper[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{upper}$')
        
        # Setup frequency index subplot
        self.ax2.set_xlabel('FI', fontsize=10, color='grey')
        self.ax2.set_ylabel('')
        self.ax2.set_xlim(-0.5, 0.5)
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')
        self.ax2.set_xticks([])
        self.ax2.set_xticklabels([])
        self.ax2.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label C.
        self.ax2.text(0.02, 0.98, 'C.', transform=self.ax2.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')
        
        # Add threshold lines (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax2.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        # Setup timeseries subplot
        self.ax_fi_timeseries.set_xlabel('')  # No xlabel as requested
        self.ax_fi_timeseries.set_ylabel('FI', fontsize=10, color='grey')
        self.ax_fi_timeseries.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label D.
        self.ax_fi_timeseries.text(0.02, 0.98, 'D.', transform=self.ax_fi_timeseries.transAxes,
                                   fontsize=10, color='black', weight='bold',
                                   verticalalignment='top', horizontalalignment='left')
        
        # Add threshold lines to timeseries (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax_fi_timeseries.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        _tight_layout_safe(self.fig)
        
        # Handle waveform/spectra/FI panels (A-C)
        if example_waveform is not None:
            # Use example_waveform for panels A-C
            if isinstance(example_waveform, Trace):
                tr = example_waveform
            elif isinstance(example_waveform, Stream):
                tr = example_waveform.merge()[0]
            else:
                raise TypeError("example_waveform must be a Trace or Stream object")
            
            # Create temporary stream_info for example_waveform
            example_stream_info = {
                'trace': tr,
                'color': 'auto',
                'analysis_window': self.analysis_window,
                'magnitude': None
            }
            self._plot_stream(example_stream_info)
        else:
            # Use stored streams (backward compatibility)
            for stream_info in self._streams:
                self._plot_stream(stream_info)
        
        # Handle timeseries panel (D)
        if catalog is not None and event_traces is not None:
            # Extract data from catalog
            times, fi_values, mags = self.extract_catalog_data(
                catalog, event_traces, self.Alower, self.Aupper, self.analysis_window)
            self._plot_timeseries_data(times, fi_values, mags)
        elif times is not None and fi_values is not None:
            # Use pre-computed data
            self._plot_timeseries_data(times, fi_values, magnitudes)
        else:
            # Use stored streams (backward compatibility)
            self._plot_timeseries()
        
        # Align y-labels: waveform, spectrum, and timeseries should be aligned
        self.fig.align_ylabels([self.ax_wave, self.ax1, self.ax_fi_timeseries])
        
        # Apply custom limits if set
        self._apply_limits()
        
        plt.draw()
        return self.fig

    def _plot_timeseries(self):
        """
        Internal method to plot the FI timeseries on the timeseries axis.
        This method collects FI values and times from all stored streams and plots them.
        """
        if self.ax_fi_timeseries is None:
            return
        
        # Collect FI values and times for all streams
        times = []
        fi_values = []
        colors = []
        
        for stream_info in self._streams:
            tr = stream_info['trace']
            color = stream_info['color']
            analysis_window_override = stream_info['analysis_window']
            
            # Determine which analysis_window to use
            window_to_use = analysis_window_override if analysis_window_override is not None else self.analysis_window
            
            # Calculate frequency index using static method
            freq_index = self.calculate_frequency_index(tr, self.Alower, self.Aupper, window_to_use)
            
            # Get time (use starttime of the trace)
            time = tr.stats.starttime
            
            # Determine color based on user preference
            if color == "auto":
                plot_color = self._get_color_from_freq_index(freq_index)
            else:
                plot_color = color
            
            times.append(time)
            fi_values.append(freq_index)
            colors.append(plot_color)
        
        # Plot as scatter plot (using 'k' for all points to match figure_for_margarita)
        if len(times) > 0:
            # Convert times to matplotlib dates for plotting
            time_dates = [t.matplotlib_date for t in times]
            self.ax_fi_timeseries.scatter(time_dates, fi_values, c='k', alpha=0.7, s=30, 
                                         edgecolor='k', linewidth=0.5, marker='o')
            
            # Format x-axis using ObsPy's date formatting
            _set_xaxis_obspy_dates(self.ax_fi_timeseries)

    def _plot_timeseries_data(self, times, fi_values, magnitudes=None):
        """
        Internal method to plot pre-computed FI timeseries data on the timeseries axis.
        
        Parameters:
        -----------
        times : list or array-like
            Time values. Can be UTCDateTime objects, matplotlib dates, or datetime objects.
        fi_values : list or array-like
            Frequency index values
        magnitudes : list or array-like or None, optional
            Magnitude values (currently not used for coloring, but kept for future use)
        """
        if self.ax_fi_timeseries is None:
            return
        
        if len(times) != len(fi_values):
            raise ValueError(f"times ({len(times)}) and fi_values ({len(fi_values)}) must have same length")
        
        # Determine colors based on FI values and thresholds
        colors = []
        if self.thresholds is not None and self.threshold_colors is not None:
            sorted_thresholds = sorted(self.thresholds)
            for fi in fi_values:
                color_index = 0
                for i, threshold in enumerate(sorted_thresholds):
                    if fi > threshold:
                        color_index = i + 1
                    else:
                        break
                color_index = min(color_index, len(self.threshold_colors) - 1)
                colors.append(self.threshold_colors[color_index] if self.threshold_colors[color_index] is not None else "black")
        else:
            colors = ['black'] * len(fi_values)
        
        # Convert times to matplotlib dates if needed
        if len(times) > 0:
            if isinstance(times[0], UTCDateTime):
                time_dates = [t.matplotlib_date for t in times]
            else:
                time_dates = times
            
            # Plot as scatter plot (using 'k' for all points to match figure_for_margarita)
            self.ax_fi_timeseries.scatter(time_dates, fi_values, c='k', alpha=0.7, s=30, 
                                         edgecolor='k', linewidth=0.5, marker='o')
            
            # Format x-axis using ObsPy's date formatting if UTCDateTime
            if len(times) > 0 and isinstance(times[0], UTCDateTime):
                _set_xaxis_obspy_dates(self.ax_fi_timeseries)

    def _apply_limits(self):
        """
        Apply custom axis limits if they have been set.
        Also sets default waveform xlim to data extent in seconds.
        """
        # Apply waveform xlim
        if self.ax_wave is not None:
            if self._wave_xlim is not None:
                self.ax_wave.set_xlim(self._wave_xlim)
            else:
                # Default: find max waveform length across all streams and example_waveform
                max_length = 0
                # Check stored streams
                for stream_info in self._streams:
                    tr = stream_info['trace']
                    length = len(tr.data) / tr.stats.sampling_rate
                    max_length = max(max_length, length)
                # Check example_waveform if it was used (stored in a temporary attribute)
                if hasattr(self, '_example_waveform_length'):
                    max_length = max(max_length, self._example_waveform_length)
                # Set xlim to data extent (0 to max_length in seconds)
                # This should be the length of the waveform in seconds
                if max_length > 0:
                    self.ax_wave.set_xlim(0, max_length)
                else:
                    # Fallback: get actual data extent from plotted lines
                    if len(self.ax_wave.lines) > 0:
                        # Get x-data from all lines and find max
                        all_xdata = []
                        for line in self.ax_wave.lines:
                            xdata = line.get_xdata()
                            if len(xdata) > 0:
                                all_xdata.extend(xdata)
                        if len(all_xdata) > 0:
                            max_length = max(all_xdata)
                            self.ax_wave.set_xlim(0, max_length)
        
        # Apply spectra xlim (frequency axis)
        if self.ax1 is not None and self._spectra_xlim is not None:
            self.ax1.set_xlim(self._spectra_xlim)
        
        # Apply spectra ylim (amplitude axis)
        if self.ax1 is not None and self._spectra_ylim is not None:
            self.ax1.set_ylim(self._spectra_ylim)
        
        # Apply frequency index ylim
        if self.ax2 is not None and self._fi_ylim is not None:
            self.ax2.set_ylim(self._fi_ylim)

    def set_wave_xlim(self, limits):
        """
        Set waveform x-axis limits (in seconds).
        
        Parameters:
        -----------
        limits : list or tuple
            [xmin, xmax] in seconds. Consistent with matplotlib's set_xlim().
        """
        if len(limits) != 2:
            raise ValueError("limits must be a 2-element list or tuple [xmin, xmax]")
        self._wave_xlim = list(limits)
        # Apply immediately if figure exists
        if self.ax_wave is not None:
            self.ax_wave.set_xlim(self._wave_xlim)

    def set_spectra_xlim(self, limits):
        """
        Set spectra frequency x-axis limits (in Hz).
        
        Parameters:
        -----------
        limits : list or tuple
            [xmin, xmax] in Hz. Consistent with matplotlib's set_xlim().
        """
        if len(limits) != 2:
            raise ValueError("limits must be a 2-element list or tuple [xmin, xmax]")
        self._spectra_xlim = list(limits)
        # Apply immediately if figure exists
        if self.ax1 is not None:
            self.ax1.set_xlim(self._spectra_xlim)

    def set_spectra_ylim(self, limits):
        """
        Set spectra amplitude y-axis limits.
        
        Parameters:
        -----------
        limits : list or tuple
            [ymin, ymax]. Consistent with matplotlib's set_ylim().
        """
        if len(limits) != 2:
            raise ValueError("limits must be a 2-element list or tuple [ymin, ymax]")
        self._spectra_ylim = list(limits)
        # Apply immediately if figure exists
        if self.ax1 is not None:
            self.ax1.set_ylim(self._spectra_ylim)

    def set_fi_ylim(self, limits):
        """
        Set frequency index y-axis limits.
        
        Parameters:
        -----------
        limits : list or tuple
            [ymin, ymax]. Consistent with matplotlib's set_ylim().
        """
        if len(limits) != 2:
            raise ValueError("limits must be a 2-element list or tuple [ymin, ymax]")
        self._fi_ylim = list(limits)
        # Apply immediately if figure exists
        if self.ax2 is not None:
            self.ax2.set_ylim(self._fi_ylim)

    def add_stream(self, tr, analysis_window=None, color="auto", magnitude=None):
        """
        Add seismic stream data to be plotted.

        Parameters:
        -----------
        tr : obspy.Trace or Stream
            ObsPy Trace or Stream object. If Stream, will be merged and the first Trace will be used.
        analysis_window : list of float or None, optional
            Time window in seconds [start, end] to use for spectrum and frequency
            index calculation. If None, uses the class default (self.analysis_window).
            This parameter overrides the class default if provided.
        color : str, default="auto"
            Color for plotting the spectrum and frequency index point.
            If "auto", uses color determined by frequency index and thresholds
            (requires thresholds and threshold_colors to be set).
            Otherwise, uses the specified color (e.g., "red", "blue", "#FF0000").
        magnitude : float or None, optional
            Magnitude value associated with this stream. Used for FI vs Magnitude plots.
            If None, this stream will not be included in magnitude plots.
        """
        if isinstance(tr, Trace):
            tr = tr
        elif isinstance(tr, Stream):
            # Collect the first Trace from the Stream
            tr = tr.merge()[0]
        else:
            raise TypeError("Input must be a Trace or Stream object")

        # Store the trace with its metadata
        self._streams.append({
            'trace': tr,
            'color': color,
            'analysis_window': analysis_window,
            'magnitude': magnitude
        })

    def add_streams_with_magnitudes(self, streams, magnitudes, analysis_window=None, color="auto"):
        """
        Add multiple seismic streams with their corresponding magnitudes.
        
        This is a convenience method for adding multiple streams when you have
        magnitudes as a list. The magnitudes should be in the same order as the streams.
        
        Parameters:
        -----------
        streams : list of obspy.Trace or Stream
            List of ObsPy Trace or Stream objects to add.
        magnitudes : list of float
            List of magnitude values corresponding to each stream, in the same order.
            Must have the same length as streams.
        analysis_window : list of float or None, optional
            Time window in seconds [start, end] to use for spectrum and frequency
            index calculation. Applied to all streams.
        color : str, default="auto"
            Color for plotting. Applied to all streams.
        """
        if len(streams) != len(magnitudes):
            raise ValueError(f"Number of streams ({len(streams)}) must match number of magnitudes ({len(magnitudes)})")
        
        for tr, mag in zip(streams, magnitudes):
            self.add_stream(tr, analysis_window=analysis_window, color=color, magnitude=mag)

    def add_catalog(self, times, fi_values, magnitudes=None, **scatter_kwargs):
        """
        Add a catalog dataset for plotting on timeseries and FI vs Magnitude panels.
        
        This method stores catalog data that will be plotted when plot_all() is called.
        Multiple catalogs can be added, each with different styling (markers, colors, etc.).
        
        Parameters:
        -----------
        times : list or array-like
            Time values. Can be UTCDateTime objects, matplotlib dates, or datetime objects.
            Must have the same length as fi_values.
        fi_values : list or array-like
            Frequency index values. Must have the same length as times.
        magnitudes : list or array-like or None, optional
            Magnitude values. If None, this catalog won't be plotted on FI vs Magnitude panel.
            If provided, must have the same length as times and fi_values.
        **scatter_kwargs : dict
            Any keyword arguments passed to matplotlib scatter().
            Common ones: marker, facecolor, edgecolor, linewidth, alpha, s, label, etc.
            Note: Use 'facecolor' (not 'color' or 'c') to ensure edgecolor works properly.
            Defaults: marker='o', facecolor='k', edgecolor='k', linewidth=0.5, alpha=0.3, s=30
            
        Examples:
        ---------
        >>> buurman.add_catalog(times, fi_values, magnitudes, marker='o', facecolor='k', label='Original Catalog')
        >>> buurman.add_catalog(times2, fi_values2, magnitudes2, marker='s', facecolor='none', 
        ...                     edgecolor='k', linewidth=2.5, label='Catalog LPs')
        """
        if len(times) != len(fi_values):
            raise ValueError(f"times ({len(times)}) and fi_values ({len(fi_values)}) must have same length")
        
        if magnitudes is not None and len(magnitudes) != len(times):
            raise ValueError(f"magnitudes ({len(magnitudes)}) must have same length as times ({len(times)})")
        
        # Merge default scatter kwargs with user-provided ones (user overrides defaults)
        scatter_params = self._default_scatter_kwargs.copy()
        scatter_params.update(scatter_kwargs)
        
        # Handle 'c'/'color' vs 'facecolor' - prefer facecolor for clarity with edgecolor
        if 'color' in scatter_params:
            if 'facecolor' not in scatter_params:
                scatter_params['facecolor'] = scatter_params.pop('color')
            else:
                scatter_params.pop('color')  # facecolor takes precedence
        if 'c' in scatter_params:
            if 'facecolor' not in scatter_params:
                scatter_params['facecolor'] = scatter_params.pop('c')
            else:
                scatter_params.pop('c')  # facecolor takes precedence
        
        # Store catalog data
        self._catalogs.append({
            'times': list(times),
            'fi_values': list(fi_values),
            'magnitudes': list(magnitudes) if magnitudes is not None else None,
            'scatter_kwargs': scatter_params
        })

    def _calculate_fi_for_stream(self, stream_info):
        """
        Calculate frequency index for a stream without plotting.
        
        Parameters:
        -----------
        stream_info : dict
            Dictionary containing 'trace', 'color', and 'analysis_window' keys.
            
        Returns:
        --------
        float or None
            Frequency index value, or None if calculation fails.
        """
        tr = stream_info['trace']
        analysis_window_override = stream_info['analysis_window']
        
        # Determine which analysis_window to use
        window_to_use = analysis_window_override if analysis_window_override is not None else self.analysis_window
        
        # Prepare data for spectrum calculation
        if window_to_use is not None:
            if len(window_to_use) != 2:
                raise ValueError("analysis_window must be a 2-element list [start, end]")
            start_time, end_time = window_to_use
            sampling_rate = tr.stats.sampling_rate
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            end_sample = min(end_sample, len(tr.data))
            tr_for_analysis = tr.copy()
            tr_for_analysis.data = tr.data[start_sample:end_sample]
        else:
            tr_for_analysis = tr
        
        # Calculate power spectral density using Welch's method
        # Use full data length to match FFT resolution (as in Buurman & West 2010)
        sampling_rate = tr_for_analysis.stats.sampling_rate
        nperseg = len(tr_for_analysis.data)
        
        freqs, psd = welch(tr_for_analysis.data, fs=sampling_rate, nperseg=nperseg,
                           scaling='density', detrend='linear')
        
        # Remove DC component
        freqs = freqs[1:]
        psd = psd[1:]
        
        # Convert PSD (power) to amplitude spectrum by taking square root
        # This matches the FFT-based approach used by Buurman & West (2010)
        amplitude = np.sqrt(psd)
        
        # Normalize by area under curve
        area = np.trapezoid(amplitude, freqs)
        amplitude_normalized = amplitude / area
        
        # Calculate frequency index
        freq_index = self._calculate_frequency_index(freqs, amplitude_normalized)
        return freq_index

    def _plot_stream(self, stream_info, waveform_y_position=None):
        """
        Internal method to plot a single stored stream on the current figure.
        
        Parameters:
        -----------
        stream_info : dict
            Dictionary containing 'trace', 'color', and 'analysis_window' keys.
        waveform_y_position : float or None, optional
            Y-position for waveform when using "even" spacing mode. If None,
            uses spacing mode to determine position.
        """
        tr = stream_info['trace']
        color = stream_info['color']
        analysis_window_override = stream_info['analysis_window']

        # Determine which analysis_window to use (method parameter overrides class default)
        window_to_use = analysis_window_override if analysis_window_override is not None else self.analysis_window

        # Prepare data for spectrum calculation
        if window_to_use is not None:
            # Trim waveform to analysis_window
            if len(window_to_use) != 2:
                raise ValueError("analysis_window must be a 2-element list [start, end]")
            start_time, end_time = window_to_use
            sampling_rate = tr.stats.sampling_rate
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            # Ensure we don't exceed data length
            end_sample = min(end_sample, len(tr.data))
            tr_for_analysis = tr.copy()
            tr_for_analysis.data = tr.data[start_sample:end_sample]
        else:
            # Use entire waveform
            tr_for_analysis = tr

        # Calculate power spectral density using Welch's method
        # Use full data length to match FFT resolution (as in Buurman & West 2010)
        sampling_rate = tr_for_analysis.stats.sampling_rate
        nperseg = len(tr_for_analysis.data)

        freqs, psd = welch(tr_for_analysis.data, fs=sampling_rate, nperseg=nperseg,
                           scaling='density', detrend='linear')

        # Remove DC component
        freqs = freqs[1:]
        psd = psd[1:]

        # Convert PSD (power) to amplitude spectrum by taking square root
        # This matches the FFT-based approach used by Buurman & West (2010)
        amplitude = np.sqrt(psd)

        # Normalize by area under curve
        area = np.trapezoid(amplitude, freqs)
        amplitude_normalized = amplitude / area

        # Calculate frequency index (if we have spectrum/freq index axes, or if using FI spacing in fig2)
        freq_index = None
        if self.ax1 is not None or self.ax2 is not None:
            freq_index = self._calculate_frequency_index(freqs, amplitude_normalized)
        elif self.ax_wave is not None and self._waveform_spacing_mode == "fi":
            # For fig2 with FI spacing, calculate FI even without spectrum/FI axes
            freq_index = self._calculate_frequency_index(freqs, amplitude_normalized)

        # Determine color based on user preference
        if color == "auto":
            if freq_index is not None:
                plot_color = self._get_color_from_freq_index(freq_index)
            else:
                plot_color = "black"
        else:
            plot_color = color

        # Plot waveform if waveform axis exists
        if self.ax_wave is not None:
            # Calculate waveform length in seconds (data extent)
            waveform_length_seconds = len(tr.data) / tr.stats.sampling_rate
            
            normalized_data = tr.data / np.max(np.abs(tr.data))
            time_axis = np.arange(len(normalized_data)) / tr.stats.sampling_rate
            # time_axis goes from 0 to waveform_length_seconds
            
            # Determine y-offset based on spacing mode
            if self.ax1 is not None or self.ax2 is not None:
                # For fig23 layout, always use FI spacing
                if freq_index is not None:
                    self._waveform_freq_indices.append(freq_index)
                    waveform_y = normalized_data + freq_index
                    self.ax_wave.plot(time_axis, waveform_y, color='k', linewidth=1)
                    
                    # Set y-axis limits based on frequency indices (with some padding)
                    if len(self._waveform_freq_indices) > 0:
                        min_fi = min(self._waveform_freq_indices)
                        max_fi = max(self._waveform_freq_indices)
                        # Add padding: waveform amplitude range is roughly -1 to 1, so add 1.5 on each side
                        y_padding = 1.5
                        self.ax_wave.set_ylim(min_fi - y_padding, max_fi + y_padding)
            else:
                # For fig2 layout, use spacing mode
                if self._waveform_spacing_mode == "fi":
                    # Space by frequency index - need to calculate FI
                    if freq_index is None:
                        # Calculate FI even though we don't have spectrum/FI axes
                        freq_index = self._calculate_fi_for_stream(stream_info)
                    if freq_index is not None:
                        self._waveform_freq_indices.append(freq_index)
                        waveform_y = normalized_data + freq_index
                        self.ax_wave.plot(time_axis, waveform_y, color='k', linewidth=1)
                    else:
                        # Fallback to center if FI calculation fails
                        waveform_y = normalized_data + self._waveform_spacing_center
                        self.ax_wave.plot(time_axis, waveform_y, color='k', linewidth=1)
                elif self._waveform_spacing_mode == "even":
                    # Space evenly - use provided y_position
                    if waveform_y_position is not None:
                        waveform_y = normalized_data + waveform_y_position
                        self.ax_wave.plot(time_axis, waveform_y, color='k', linewidth=1)
                    else:
                        # Fallback to center if position not provided
                        waveform_y = normalized_data + self._waveform_spacing_center
                        self.ax_wave.plot(time_axis, waveform_y, color='k', linewidth=1)
                else:  # "center" mode
                    # All at center value
                    waveform_y = normalized_data + self._waveform_spacing_center
                    self.ax_wave.plot(time_axis, waveform_y, color='k', linewidth=1)
            
            # xlim will be set in _apply_limits() after all streams are plotted
            # ylim will be set after all streams are plotted (see plot_fig2)
            
            # No y-ticks
            self.ax_wave.set_yticks([])
            
            # Highlight analysis_window region if specified (only add once using class-level analysis_window)
            if not self._waveform_highlight_added and self.analysis_window is not None:
                if len(self.analysis_window) != 2:
                    raise ValueError("analysis_window must be a 2-element list [start, end]")
                start_time, end_time = self.analysis_window
                # Highlight spans full y-range (using 'k' to match figure_for_margarita)
                ylim = self.ax_wave.get_ylim()
                self.ax_wave.axvspan(start_time, end_time, ymin=0, ymax=1,
                                     alpha=self.highlight_alpha,
                                     color='k', zorder=0)
                self._waveform_highlight_added = True

        # Plot the spectrum if spectrum axis exists (using 'k' to match figure_for_margarita)
        if self.ax1 is not None:
            self.ax1.plot(freqs, amplitude_normalized, color='k', linewidth=1.5,
                            label=f'{tr.stats.station}.{tr.stats.channel}')

        # Plot frequency index if frequency index axis exists (using 'k' to match figure_for_margarita)
        if self.ax2 is not None and freq_index is not None:
            self.ax2.scatter(0, freq_index, c='k', alpha=0.7, s=30, edgecolor='k', linewidth=0.5)

        # Align labels (matching notebook style)
        if self.ax_wave is not None and self.ax1 is not None:
            # Align y-axis labels on the left (waveform and spectrum)
            self.fig.align_ylabels([self.ax_wave, self.ax1])
        if self.ax1 is not None and self.ax2 is not None:
            # Align x-axis labels at the bottom (spectrum and frequency index)
            self.fig.align_xlabels([self.ax1, self.ax2])

    def _get_color_from_freq_index(self, freq_index):
        """
        Determine color based on frequency index value and thresholds.

        Parameters:
        -----------
        freq_index : float
            Calculated frequency index value

        Returns:
        --------
        str : Color for plotting (defaults to "black" if thresholds/colors not set)
        """
        # Handle None cases - return black if thresholds or colors not set
        if self.thresholds is None or self.threshold_colors is None:
            return "black"
        
        # Sort thresholds to handle them in order
        sorted_thresholds = sorted(self.thresholds)

        # Determine which region the frequency index falls into
        color_index = 0  # Default to first color

        for i, threshold in enumerate(sorted_thresholds):
            if freq_index > threshold:
                color_index = i + 1
            else:
                break

        # Ensure we don't exceed available colors
        color_index = min(color_index, len(self.threshold_colors) - 1)

        # Handle None elements in threshold_colors
        selected_color = self.threshold_colors[color_index]
        if selected_color is None:
            return "black"
        
        return selected_color

    def _calculate_frequency_index(self, freqs, amplitude_spectrum):
        """
        Calculate frequency index based on spectral content in defined bands.

        This is a simplified version - you may need to adjust based on
        the specific definition used in Buurman et al.
        """
        # Find indices for frequency bands
        lower_mask = (freqs >= self.Alower[0]) & (freqs < self.Alower[1])  # create mask
        upper_mask = (freqs >= self.Aupper[0]) & (freqs < self.Aupper[1])  # create mask

        # Calculate frequency index (log ratio)
        # freq_index = np.log10(np.mean(amplitude_spectrum[upper_mask].sum(axis=1)) / np.mean(amplitude_spectrum[lower_mask].sum(axis=1)))
        freq_index = np.log10(np.mean(amplitude_spectrum[upper_mask]) / np.mean(amplitude_spectrum[lower_mask]))

        return freq_index

    def _plot_catalogs_timeseries(self):
        """
        Plot all stored catalogs on the timeseries axis.
        If magnitudes are available, scatter sizes will be calculated from magnitudes.
        """
        if self.ax_fi_timeseries is None or len(self._catalogs) == 0:
            return
        
        for cat in self._catalogs:
            times = cat['times']
            fi_values = cat['fi_values']
            magnitudes = cat['magnitudes']
            scatter_kwargs = cat['scatter_kwargs'].copy()
            
            # Convert times to matplotlib dates if needed
            if len(times) > 0:
                if isinstance(times[0], UTCDateTime):
                    time_dates = [t.matplotlib_date for t in times]
                else:
                    time_dates = times
                
                # Always calculate scatter sizes from magnitudes if available (same as fi_mag plot)
                # Override any explicit 's' to ensure consistency
                if magnitudes is not None:
                    scatter_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitudes]
                    scatter_kwargs['s'] = scatter_sizes
                
                # Ensure facecolor is used (not color/c) so edgecolor works properly
                if 'facecolor' not in scatter_kwargs and 'c' not in scatter_kwargs and 'color' not in scatter_kwargs:
                    scatter_kwargs['facecolor'] = 'k'
                
                self.ax_fi_timeseries.scatter(time_dates, fi_values, **scatter_kwargs)
        
        # Format x-axis using ObsPy's date formatting if needed
        if len(self._catalogs) > 0 and len(self._catalogs[0]['times']) > 0:
            if isinstance(self._catalogs[0]['times'][0], UTCDateTime):
                _set_xaxis_obspy_dates(self.ax_fi_timeseries)
        
        # No legend for timeseries plot (as per user request)

    def _plot_catalogs_fi_magnitude(self):
        """
        Plot all stored catalogs on the FI vs Magnitude axis.
        Only catalogs with magnitudes will be plotted.
        """
        if self.ax_fi_mag is None or len(self._catalogs) == 0:
            return
        
        all_magnitudes = []
        legend_handles = []
        
        for cat in self._catalogs:
            if cat['magnitudes'] is None:
                continue
            
            times = cat['times']
            fi_values = cat['fi_values']
            magnitudes = cat['magnitudes']
            scatter_kwargs = cat['scatter_kwargs'].copy()
            
            # Filter out any None or NaN values
            valid_indices = [i for i in range(len(fi_values)) 
                           if fi_values[i] is not None and magnitudes[i] is not None 
                           and not (np.isnan(fi_values[i]) or np.isnan(magnitudes[i]))]
            
            if len(valid_indices) == 0:
                continue
            
            fi_values_valid = [fi_values[i] for i in valid_indices]
            magnitudes_valid = [magnitudes[i] for i in valid_indices]
            all_magnitudes.extend(magnitudes_valid)
            
            # Calculate scatter sizes based on magnitude
            scatter_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitudes_valid]
            
            # Override 's' with calculated sizes, but preserve other kwargs
            scatter_kwargs['s'] = scatter_sizes
            
            # Ensure facecolor is used (not color/c) so edgecolor works properly
            if 'facecolor' not in scatter_kwargs and 'c' not in scatter_kwargs and 'color' not in scatter_kwargs:
                scatter_kwargs['facecolor'] = 'k'
            
            # Plot this catalog
            self.ax_fi_mag.scatter(fi_values_valid, magnitudes_valid, **scatter_kwargs)
            
            # Add to legend if label provided
            if 'label' in scatter_kwargs and scatter_kwargs['label']:
                legend_kwargs = scatter_kwargs.copy()
                legend_kwargs['s'] = 100  # Fixed size for legend
                legend_handles.append(
                    self.ax_fi_mag.scatter([], [], **legend_kwargs)
                )
        
        # Set y-axis limits based on all magnitudes
        if len(all_magnitudes) > 0:
            mag_min = min(all_magnitudes)
            mag_max = max(all_magnitudes)
            mag_range = mag_max - mag_min
            if mag_range > 0:
                y_margin = mag_range * 0.1
                self.ax_fi_mag.set_ylim(mag_min - y_margin, mag_max + y_margin)
            else:
                self.ax_fi_mag.set_ylim(mag_min - 0.5, mag_max + 0.5)
        
        # Create magnitude size legend
        if len(all_magnitudes) > 0:
            mag_range_legend = max(all_magnitudes) - min(all_magnitudes)
            if mag_range_legend > 0:
                magnitude_levels = np.linspace(min(all_magnitudes), max(all_magnitudes), 5)
            else:
                magnitude_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            legend_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitude_levels]
            legend_labels = [f'M{mag:.1f}' for mag in magnitude_levels]
            
            for size, label in zip(legend_sizes, legend_labels):
                legend_handles.append(
                    self.ax_fi_mag.scatter([], [], s=size, c='gray', alpha=0.7,
                                         edgecolor='black', linewidth=0.5, marker='o', label=label)
                )
        
        # Add combined legend to top right
        if len(legend_handles) > 0:
            self.ax_fi_mag.legend(handles=legend_handles, loc='upper right',
                                 fontsize=9, frameon=True)

    def plot_all(self, figsize=(12, 8), example_waveform=None,
                 title_waveform=None, title_timeseries=None):
        """
        Create combined figure with all panels: waveform/spectra/FI, FI vs Magnitude, and timeseries.
        
        This method creates a layout matching figure_for_margarita.ipynb:
        - Upper left: Waveform, spectra, and frequency index (panels A-C)
        - Upper right: FI vs Magnitude (panel D)
        - Bottom: Timeseries of FI over time (panel E, spans full width)
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size (width, height) in inches.
        example_waveform : obspy.Trace or None, optional
            Trace to display in waveform, spectra, and FI panels (panels A-C).
            If None, uses stored streams.
        title_waveform : str or None, optional
            Title for the waveform panel. If None, no title is added.
        title_timeseries : str or None, optional
            Title for the timeseries panel. If None, no title is added.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
            
        Notes:
        ------
        Catalogs should be added using add_catalog() before calling this method.
        The catalogs will be plotted on both the timeseries panel (E) and FI vs Magnitude panel (D).
        """
        # Reset state
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        # Create figure with 2x2 grid layout
        self.fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[1.5, 1], width_ratios=[1, 1],
                               hspace=0.3, wspace=0.3)
        
        # ========== Upper left: Buurman_fig23 (nested GridSpec) ==========
        # Create a nested GridSpec within the top left cell for Buurman_fig23 layout
        gs_buurman = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0],
                                                       height_ratios=[2.5, 5], width_ratios=[7, 1],
                                                       hspace=0.3, wspace=0.2)
        
        # Top: Waveform (spans both columns)
        self.ax_wave = self.fig.add_subplot(gs_buurman[0, :])
        self.ax_wave.set_ylabel('AMPLITUDE', fontsize=10, color='grey')
        self.ax_wave.set_xlabel('')
        self.ax_wave.set_yticks([])
        self.ax_wave.tick_params(axis='both', colors='grey', labelsize=10)
        if title_waveform is not None:
            self.ax_wave.set_title(title_waveform, fontsize=10, color='grey')
        # Add panel label A.
        self.ax_wave.text(0.02, 0.98, 'A.', transform=self.ax_wave.transAxes,
                         fontsize=10, color='black', weight='bold',
                         verticalalignment='top', horizontalalignment='left')
        
        # Bottom left: Spectrum
        self.ax1 = self.fig.add_subplot(gs_buurman[1, 0])
        self.ax1.set_xlabel('FREQUENCY (HZ)', fontsize=10, color='grey')
        self.ax1.set_ylabel('AMPLITUDE, NORMALIZED', fontsize=10, color='grey')
        self.ax1.set_xscale('log')
        self.ax1.set_xlim(0.1, 50)
        self.ax1.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label B.
        self.ax1.text(0.02, 0.98, 'B.', transform=self.ax1.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')
        
        # Add shaded regions for frequency bands (using 'k' to match figure_for_margarita)
        if len(self.Alower) >= 1 and len(self.Aupper) >= 1:
            self.ax1.axvspan(self.Alower[0], self.Alower[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{lower}$')
        if len(self.Alower) >= 2 and len(self.Aupper) >= 2:
            self.ax1.axvspan(self.Aupper[0], self.Aupper[1], alpha=self.highlight_alpha,
                             color='k', label='$A_{upper}$')
        
        # Bottom right: Frequency Index
        self.ax2 = self.fig.add_subplot(gs_buurman[1, 1])
        self.ax2.set_xlabel('FI', fontsize=10, color='grey')
        self.ax2.set_ylabel('')
        self.ax2.set_xlim(-0.5, 0.5)
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')
        self.ax2.set_xticks([])
        self.ax2.set_xticklabels([])
        self.ax2.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label C.
        self.ax2.text(0.02, 0.98, 'C.', transform=self.ax2.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')
        
        # Add threshold lines (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax2.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        # ========== Upper right: FI vs Magnitude ==========
        self.ax_fi_mag = self.fig.add_subplot(gs[0, 1])
        self.ax_fi_mag.set_ylabel('MAGNITUDE', fontsize=10, color='grey')
        self.ax_fi_mag.set_xlabel('FI', fontsize=10, color='grey')
        self.ax_fi_mag.set_title('FI VS MAGNITUDE', fontsize=10, color='grey')
        self.ax_fi_mag.grid(True, alpha=0.3)
        self.ax_fi_mag.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label D.
        self.ax_fi_mag.text(0.02, 0.98, 'D.', transform=self.ax_fi_mag.transAxes,
                           fontsize=10, color='black', weight='bold',
                           verticalalignment='top', horizontalalignment='left')
        
        # ========== Bottom row: Time Series of FI (spans both columns) ==========
        self.ax_fi_timeseries = self.fig.add_subplot(gs[1, :])
        self.ax_fi_timeseries.set_xlabel('TIME', fontsize=10, color='grey')
        self.ax_fi_timeseries.set_ylabel('FI', fontsize=10, color='grey')
        if title_timeseries is not None:
            self.ax_fi_timeseries.set_title(title_timeseries, fontsize=10, color='grey')
        self.ax_fi_timeseries.grid(True, alpha=0.3)
        self.ax_fi_timeseries.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label E.
        self.ax_fi_timeseries.text(0.02, 0.98, 'E.', transform=self.ax_fi_timeseries.transAxes,
                                   fontsize=10, color='black', weight='bold',
                                   verticalalignment='top', horizontalalignment='left')
        
        # Add threshold lines to timeseries (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                self.ax_fi_timeseries.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        _tight_layout_safe(self.fig)
        
        # Handle waveform/spectra/FI panels (A-C)
        if example_waveform is not None:
            # Use example_waveform for panels A-C
            if isinstance(example_waveform, Trace):
                tr = example_waveform
            elif isinstance(example_waveform, Stream):
                tr = example_waveform.merge()[0]
            else:
                raise TypeError("example_waveform must be a Trace or Stream object")
            
            # Create temporary stream_info for example_waveform
            example_stream_info = {
                'trace': tr,
                'color': 'auto',
                'analysis_window': self.analysis_window,
                'magnitude': None
            }
            self._plot_stream(example_stream_info)
        else:
            # Use stored streams (backward compatibility)
            for stream_info in self._streams:
                self._plot_stream(stream_info)
        
        # Handle FI vs Magnitude panel (D) - plot all catalogs
        self._plot_catalogs_fi_magnitude()
        
        # Handle timeseries panel (E) - plot all catalogs
        self._plot_catalogs_timeseries()
        
        # Align y-labels: waveform, spectrum, and timeseries should be aligned
        self.fig.align_ylabels([self.ax_wave, self.ax1, self.ax_fi_timeseries])
        
        # Apply custom limits if set
        self._apply_limits()
        
        plt.draw()
        return self.fig

    def plot_fi_magnitude(self, figsize=(6, 6)):
        """
        Create FI vs Magnitude scatter plot.
        
        This method creates a scatter plot with FI on the x-axis and Magnitude on the y-axis.
        The scatter point sizes are proportional to magnitude. Only streams with magnitude
        values will be included in the plot.
        
        Parameters:
        -----------
        figsize : tuple, default=(6, 6)
            Figure size (width, height) in inches.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        # Filter streams that have magnitude values
        streams_with_mag = [s for s in self._streams if s['magnitude'] is not None]
        
        if len(streams_with_mag) == 0:
            raise ValueError("No streams with magnitude values found. Provide magnitude values using add_stream(magnitude=...).")
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        self.ax_fi_mag = self.fig.add_subplot(1, 1, 1)
        
        # Setup axis labels and formatting (matching notebook style)
        self.ax_fi_mag.set_ylabel('MAGNITUDE', fontsize=10, color='grey')
        self.ax_fi_mag.set_xlabel('FI', fontsize=10, color='grey')
        self.ax_fi_mag.set_title('FI VS MAGNITUDE', fontsize=10, color='grey')
        self.ax_fi_mag.grid(True, alpha=0.3)
        self.ax_fi_mag.tick_params(axis='both', colors='grey', labelsize=10)
        
        # Collect FI values and magnitudes
        fi_values = []
        magnitudes = []
        colors = []
        
        for stream_info in streams_with_mag:
            # Calculate FI for this stream
            fi = self._calculate_fi_for_stream(stream_info)
            if fi is not None:
                fi_values.append(fi)
                magnitudes.append(stream_info['magnitude'])
                
                # Determine color
                color = stream_info['color']
                if color == "auto":
                    plot_color = self._get_color_from_freq_index(fi)
                else:
                    plot_color = color
                colors.append(plot_color)
        
        if len(fi_values) == 0:
            raise ValueError("Could not calculate frequency index for any streams with magnitude values.")
        
        # Calculate scatter sizes based on magnitude (matching notebook formula)
        scatter_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitudes]
        
        # Plot scatter points (matching notebook style - all lightgray)
        self.ax_fi_mag.scatter(fi_values, magnitudes, s=scatter_sizes,
                               alpha=0.7, facecolor='lightgray', edgecolor='black',
                               linewidth=0.5, marker='o')
        
        # Set y-axis limits (with some padding)
        mag_min = min(magnitudes)
        mag_max = max(magnitudes)
        mag_range = mag_max - mag_min
        if mag_range > 0:
            y_margin = mag_range * 0.1
            self.ax_fi_mag.set_ylim(mag_min - y_margin, mag_max + y_margin)
        else:
            # If all magnitudes are the same, add some padding
            self.ax_fi_mag.set_ylim(mag_min - 0.5, mag_max + 0.5)
        
        # Create legend with magnitude levels (matching notebook style)
        # Determine magnitude range for legend
        mag_range_legend = mag_max - mag_min
        if mag_range_legend > 0:
            # Create 5 levels across the range
            magnitude_levels = np.linspace(mag_min, mag_max, 5)
        else:
            # Default levels if all magnitudes are the same
            magnitude_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        legend_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitude_levels]
        legend_labels = [f'M{mag:.1f}' for mag in magnitude_levels]
        
        # Create legend handles
        legend_handles = []
        for size, label in zip(legend_sizes, legend_labels):
            # Create empty scatter points for legend
            legend_handles.append(self.ax_fi_mag.scatter([], [], s=size, c='gray', alpha=0.7,
                                                        edgecolor='black', linewidth=0.5, marker='o', label=label))
        
        # Add legend to top right
        if len(legend_handles) > 0:
            self.ax_fi_mag.legend(handles=legend_handles, loc='upper right',
                                 fontsize=9, frameon=True)
        
        _tight_layout_safe(self.fig)
        plt.draw()
        return self.fig

    @staticmethod
    def plot_fi_timeseries_static(times, fi_values, thresholds=None, threshold_colors=None,
                                  figsize=(8, 6), magnitudes=None):
        """
        Plot FI timeseries from pre-computed data (static method, no computation needed).
        
        Parameters:
        -----------
        times : list or array-like
            Time values. Can be:
            - list of obspy.UTCDateTime objects
            - list of matplotlib date numbers
            - list of datetime objects
        fi_values : list or array-like
            Pre-computed frequency index values
        thresholds : list of float or None, optional
            Thresholds for FI-based classification. Default: None
        threshold_colors : list of str or None, optional
            Colors for each classification region. Must have one more element than thresholds.
        figsize : tuple, default=(8, 6)
            Figure size (width, height) in inches.
        magnitudes : list or array-like or None, optional
            Magnitude values for color coding. If provided, will be used to determine colors
            if threshold_colors is set.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        
        # Setup axis labels
        ax.set_xlabel('TIME', fontsize=10, color='grey')
        ax.set_ylabel('FI', fontsize=10, color='grey')
        ax.tick_params(axis='both', colors='grey', labelsize=10)
        
        # Add threshold lines (if thresholds are set)
        if thresholds is not None:
            for threshold in thresholds:
                ax.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        # Determine colors
        if threshold_colors is not None and thresholds is not None and len(fi_values) > 0:
            # Color based on FI values and thresholds
            colors = []
            sorted_thresholds = sorted(thresholds)
            for fi in fi_values:
                color_index = 0
                for i, threshold in enumerate(sorted_thresholds):
                    if fi > threshold:
                        color_index = i + 1
                    else:
                        break
                color_index = min(color_index, len(threshold_colors) - 1)
                colors.append(threshold_colors[color_index] if threshold_colors[color_index] is not None else "black")
        else:
            colors = ['black'] * len(fi_values)
        
        # Convert times to matplotlib dates if needed
        if len(times) > 0:
            if isinstance(times[0], UTCDateTime):
                time_dates = [t.matplotlib_date for t in times]
            else:
                time_dates = times
            
            # Plot as scatter plot (using 'k' for all points to match figure_for_margarita)
            ax.scatter(time_dates, fi_values, c='k', alpha=0.7, s=30, 
                      edgecolor='k', linewidth=0.5, marker='o')
            
            # Format x-axis using ObsPy's date formatting if UTCDateTime
            if len(times) > 0 and isinstance(times[0], UTCDateTime):
                _set_xaxis_obspy_dates(ax)
        
        _tight_layout_safe(fig)
        plt.draw()
        return fig

    @staticmethod
    def plot_fi_magnitude_static(fi_values, magnitudes, thresholds=None, threshold_colors=None,
                                 figsize=(6, 6)):
        """
        Plot FI vs Magnitude from pre-computed data (static method, no computation needed).
        
        Parameters:
        -----------
        fi_values : list or array-like
            Pre-computed frequency index values
        magnitudes : list or array-like
            Magnitude values
        thresholds : list of float or None, optional
            Thresholds for FI-based classification. Default: None
        threshold_colors : list of str or None, optional
            Colors for each classification region. Must have one more element than thresholds.
        figsize : tuple, default=(6, 6)
            Figure size (width, height) in inches.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object.
        """
        if len(fi_values) != len(magnitudes):
            raise ValueError(f"fi_values ({len(fi_values)}) and magnitudes ({len(magnitudes)}) must have same length")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        
        # Setup axis labels and formatting
        ax.set_ylabel('MAGNITUDE', fontsize=10, color='grey')
        ax.set_xlabel('FI', fontsize=10, color='grey')
        ax.set_title('FI VS MAGNITUDE', fontsize=10, color='grey')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', colors='grey', labelsize=10)
        
        # Calculate scatter sizes based on magnitude
        scatter_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitudes]
        
        # Plot scatter points (matching notebook style - all lightgray)
        ax.scatter(fi_values, magnitudes, s=scatter_sizes,
                   alpha=0.7, facecolor='lightgray', edgecolor='black',
                   linewidth=0.5, marker='o')
        
        # Set y-axis limits (with some padding)
        mag_min = min(magnitudes)
        mag_max = max(magnitudes)
        mag_range = mag_max - mag_min
        if mag_range > 0:
            y_margin = mag_range * 0.1
            ax.set_ylim(mag_min - y_margin, mag_max + y_margin)
        else:
            ax.set_ylim(mag_min - 0.5, mag_max + 0.5)
        
        # Create legend with magnitude levels
        mag_range_legend = mag_max - mag_min
        if mag_range_legend > 0:
            magnitude_levels = np.linspace(mag_min, mag_max, 5)
        else:
            magnitude_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        legend_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitude_levels]
        legend_labels = [f'M{mag:.1f}' for mag in magnitude_levels]
        
        # Create legend handles
        legend_handles = []
        for size, label in zip(legend_sizes, legend_labels):
            legend_handles.append(ax.scatter([], [], s=size, c='gray', alpha=0.7,
                                            edgecolor='black', linewidth=0.5, marker='o', label=label))
        
        # Add legend to top right
        if len(legend_handles) > 0:
            ax.legend(handles=legend_handles, loc='upper right',
                     fontsize=9, frameon=True)
        
        _tight_layout_safe(fig)
        plt.draw()
        return fig

    def save(self, filename, dpi=300):
        """Save the figure to file."""
        if self.fig is None:
            raise ValueError("No figure has been created. Call plot_fig2(), plot_fig3(), or plot_fig23() first.")
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    def make_movie(self, filename, fps=None, figsize=(12, 8), dpi=100, duration=30, 
                   example_waveform=None, title_waveform=None, title_timeseries=None):
        """
        Create and save a movie showing all figure panels with animated timeseries and FI vs Magnitude.
        
        This method creates an animated movie with all panels from plot_all():
        - Panels A-C (waveform, spectra, frequency index): Static, shown once
        - Panel D (FI vs Magnitude): Animated, points appear progressively
        - Panel E (Timeseries): Animated, points appear progressively
        
        Parameters:
        -----------
        filename : str
            Output filename for the movie. Should have .mp4 or .gif extension.
            For .mp4, requires ffmpeg to be installed.
            For .gif, uses Pillow writer.
        fps : float or None, default=None
            Frames per second for the movie. If None, calculated to make movie duration
            equal to `duration` parameter.
        figsize : tuple, default=(12, 8)
            Figure size (width, height) in inches.
        dpi : int, default=100
            Resolution (dots per inch) for the movie frames.
        duration : float, default=30
            Target duration in seconds. Used to calculate fps if fps is None.
        example_waveform : obspy.Trace or None, optional
            Trace to display in waveform, spectra, and FI panels (panels A-C).
            If None, uses stored streams.
        title_waveform : str or None, optional
            Title for the waveform panel. If None, no title is added.
        title_timeseries : str or None, optional
            Title for the timeseries panel. If None, no title is added.
            
        Returns:
        --------
        None
            The movie is saved to the specified filename.
            
        Notes:
        ------
        - Requires catalogs to be added using add_catalog() before calling this method.
        - For MP4 format, ffmpeg must be installed on the system.
        - For GIF format, Pillow must be installed.
        - The animation shows data points appearing in chronological order.
        """
        if len(self._catalogs) == 0:
            raise ValueError("No catalogs found. Add catalogs using add_catalog() before making a movie.")
        
        print(f"Creating movie: {filename}...")
        print(f"  Format: {filename.split('.')[-1].upper()}")
        
        # Collect all data points from all catalogs and sort by time
        # Pre-compute everything once to avoid repeated lookups in animate()
        all_times = []
        all_fi_values = []
        all_catalog_indices = []  # Track which catalog each point belongs to
        all_scatter_kwargs = []
        all_magnitudes = []  # Pre-compute magnitudes to avoid searching in animate()
        
        for cat_idx, cat in enumerate(self._catalogs):
            times = cat['times']
            fi_values = cat['fi_values']
            scatter_kwargs = cat['scatter_kwargs'].copy()
            magnitudes = cat['magnitudes']
            
            # Convert times to matplotlib dates if needed
            if len(times) > 0:
                if isinstance(times[0], UTCDateTime):
                    time_dates = [t.matplotlib_date for t in times]
                else:
                    time_dates = times
                
                # Add all points from this catalog
                # Since times and fi_values are in the same order, we can use index directly
                for idx, (t, fi) in enumerate(zip(time_dates, fi_values)):
                    all_times.append(t)
                    all_fi_values.append(fi)
                    all_catalog_indices.append(cat_idx)
                    all_scatter_kwargs.append(scatter_kwargs.copy())
                    # Pre-compute magnitude if available (using same index)
                    if magnitudes is not None and idx < len(magnitudes):
                        all_magnitudes.append(magnitudes[idx])
                    else:
                        all_magnitudes.append(None)
        
        if len(all_times) == 0:
            raise ValueError("No data points found in catalogs.")
        
        # Sort all points by time
        sorted_indices = np.argsort(all_times)
        all_times = np.array([all_times[i] for i in sorted_indices])
        all_fi_values = np.array([all_fi_values[i] for i in sorted_indices])
        all_catalog_indices = np.array([all_catalog_indices[i] for i in sorted_indices])
        all_scatter_kwargs = [all_scatter_kwargs[i] for i in sorted_indices]
        all_magnitudes = [all_magnitudes[i] for i in sorted_indices]
        
        # Pre-compute catalog groupings for efficient slicing
        # Group indices by catalog for fast lookup
        catalog_groups = {}
        for cat_idx in range(len(self._catalogs)):
            catalog_groups[cat_idx] = np.where(all_catalog_indices == cat_idx)[0]
        
        # Determine number of frames (one per data point)
        n_frames = len(all_times)
        
        # Calculate fps if not provided
        if fps is None:
            fps = n_frames / duration
            print(f"  Calculated FPS: {fps:.2f} (for {duration}s duration)")
        else:
            print(f"  Using provided FPS: {fps}")
        
        print(f"  Frames: {n_frames}")
        print(f"  Estimated duration: {n_frames/fps:.1f} seconds")
        
        # Create figure with same layout as plot_all()
        # Reset state
        self._waveform_highlight_added = False
        self._waveform_freq_indices = []
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[1.5, 1], width_ratios=[1, 1],
                               hspace=0.3, wspace=0.3)
        
        # ========== Upper left: Buurman_fig23 (nested GridSpec) ==========
        # Create a nested GridSpec within the top left cell for Buurman_fig23 layout
        gs_buurman = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0],
                                                       height_ratios=[2.5, 5], width_ratios=[7, 1],
                                                       hspace=0.3, wspace=0.2)
        
        # Top: Waveform (spans both columns)
        ax_wave = fig.add_subplot(gs_buurman[0, :])
        ax_wave.set_ylabel('AMPLITUDE', fontsize=10, color='grey')
        ax_wave.set_xlabel('')
        ax_wave.set_yticks([])
        ax_wave.tick_params(axis='both', colors='grey', labelsize=10)
        if title_waveform is not None:
            ax_wave.set_title(title_waveform, fontsize=10, color='grey')
        # Add panel label A.
        ax_wave.text(0.02, 0.98, 'A.', transform=ax_wave.transAxes,
                     fontsize=10, color='black', weight='bold',
                     verticalalignment='top', horizontalalignment='left')
        
        # Bottom left: Spectrum
        ax1 = fig.add_subplot(gs_buurman[1, 0])
        ax1.set_xlabel('FREQUENCY (HZ)', fontsize=10, color='grey')
        ax1.set_ylabel('AMPLITUDE, NORMALIZED', fontsize=10, color='grey')
        ax1.set_xscale('log')
        ax1.set_xlim(0.1, 50)
        ax1.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label B.
        ax1.text(0.02, 0.98, 'B.', transform=ax1.transAxes,
                 fontsize=10, color='black', weight='bold',
                 verticalalignment='top', horizontalalignment='left')
        
        # Add shaded regions for frequency bands
        if len(self.Alower) >= 1 and len(self.Aupper) >= 1:
            ax1.axvspan(self.Alower[0], self.Alower[1], alpha=self.highlight_alpha,
                        color='k', label='$A_{lower}$')
        if len(self.Alower) >= 2 and len(self.Aupper) >= 2:
            ax1.axvspan(self.Aupper[0], self.Aupper[1], alpha=self.highlight_alpha,
                        color='k', label='$A_{upper}$')
        
        # Bottom right: Frequency Index
        ax2 = fig.add_subplot(gs_buurman[1, 1])
        ax2.set_xlabel('FI', fontsize=10, color='grey')
        ax2.set_ylabel('')
        ax2.set_xlim(-0.5, 0.5)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label C.
        ax2.text(0.02, 0.98, 'C.', transform=ax2.transAxes,
                 fontsize=10, color='black', weight='bold',
                 verticalalignment='top', horizontalalignment='left')
        
        # Add threshold lines (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                ax2.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        # ========== Upper right: FI vs Magnitude (ANIMATED) ==========
        ax_fi_mag = fig.add_subplot(gs[0, 1])
        ax_fi_mag.set_ylabel('MAGNITUDE', fontsize=10, color='grey')
        ax_fi_mag.set_xlabel('FI', fontsize=10, color='grey')
        ax_fi_mag.set_title('FI VS MAGNITUDE', fontsize=10, color='grey')
        ax_fi_mag.grid(True, alpha=0.3)
        ax_fi_mag.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label D.
        ax_fi_mag.text(0.02, 0.98, 'D.', transform=ax_fi_mag.transAxes,
                       fontsize=10, color='black', weight='bold',
                       verticalalignment='top', horizontalalignment='left')
        
        # Set axis limits for FI vs Magnitude
        mag_values = [m for m in all_magnitudes if m is not None]
        fi_values_with_mag = [all_fi_values[i] for i in range(len(all_fi_values)) if all_magnitudes[i] is not None]
        if len(mag_values) > 0:
            mag_min = min(mag_values)
            mag_max = max(mag_values)
            mag_range = mag_max - mag_min
            if mag_range > 0:
                y_margin = mag_range * 0.1
                ax_fi_mag.set_ylim(mag_min - y_margin, mag_max + y_margin)
            else:
                ax_fi_mag.set_ylim(mag_min - 0.5, mag_max + 0.5)
        
        if len(fi_values_with_mag) > 0:
            fi_min_mag = min(fi_values_with_mag)
            fi_max_mag = max(fi_values_with_mag)
            fi_range_mag = fi_max_mag - fi_min_mag
            if fi_range_mag > 0:
                x_margin = fi_range_mag * 0.1
                ax_fi_mag.set_xlim(fi_min_mag - x_margin, fi_max_mag + x_margin)
            else:
                ax_fi_mag.set_xlim(fi_min_mag - 0.5, fi_max_mag + 0.5)
        
        # ========== Bottom row: Time Series of FI (ANIMATED, spans both columns) ==========
        ax_fi_timeseries = fig.add_subplot(gs[1, :])
        ax_fi_timeseries.set_xlabel('TIME', fontsize=10, color='grey')
        ax_fi_timeseries.set_ylabel('FI', fontsize=10, color='grey')
        if title_timeseries is not None:
            ax_fi_timeseries.set_title(title_timeseries, fontsize=10, color='grey')
        ax_fi_timeseries.grid(True, alpha=0.3)
        ax_fi_timeseries.tick_params(axis='both', colors='grey', labelsize=10)
        # Add panel label E.
        ax_fi_timeseries.text(0.02, 0.98, 'E.', transform=ax_fi_timeseries.transAxes,
                              fontsize=10, color='black', weight='bold',
                              verticalalignment='top', horizontalalignment='left')
        
        # Add threshold lines to timeseries (if thresholds are set)
        if self.thresholds is not None:
            for i, threshold in enumerate(self.thresholds):
                ax_fi_timeseries.axhline(y=threshold, color="black", linestyle='--', alpha=0.7)
        
        # Set axis limits for timeseries based on all data
        if len(all_times) > 0:
            time_min = min(all_times)
            time_max = max(all_times)
            fi_min = min(all_fi_values)
            fi_max = max(all_fi_values)
            
            # Add padding
            time_range = time_max - time_min
            fi_range = fi_max - fi_min
            if time_range > 0:
                ax_fi_timeseries.set_xlim(time_min - 0.05 * time_range, time_max + 0.05 * time_range)
            else:
                ax_fi_timeseries.set_xlim(time_min - 1, time_max + 1)
            
            if fi_range > 0:
                ax_fi_timeseries.set_ylim(fi_min - 0.1 * fi_range, fi_max + 0.1 * fi_range)
            else:
                ax_fi_timeseries.set_ylim(fi_min - 0.5, fi_max + 0.5)
        
        # Format x-axis using ObsPy's date formatting if needed
        if len(all_times) > 0:
            if isinstance(self._catalogs[0]['times'][0], UTCDateTime):
                _set_xaxis_obspy_dates(ax_fi_timeseries)
        
        # Plot static panels (A-C) - waveform, spectra, FI
        # Temporarily set axes for _plot_stream
        self.ax_wave = ax_wave
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig
        
        if example_waveform is not None:
            # Use example_waveform for panels A-C
            if isinstance(example_waveform, Trace):
                tr = example_waveform
            elif isinstance(example_waveform, Stream):
                tr = example_waveform.merge()[0]
            else:
                raise TypeError("example_waveform must be a Trace or Stream object")
            
            # Create temporary stream_info for example_waveform
            example_stream_info = {
                'trace': tr,
                'color': 'auto',
                'analysis_window': self.analysis_window,
                'magnitude': None
            }
            self._plot_stream(example_stream_info)
        else:
            # Use stored streams (backward compatibility)
            for stream_info in self._streams:
                self._plot_stream(stream_info)
        
        # Align y-labels
        fig.align_ylabels([ax_wave, ax1, ax_fi_timeseries])
        
        # Apply custom limits if set
        self._apply_limits()
        
        _tight_layout_safe(fig)
        
        # Animation function - only updates panels D and E
        def animate(frame):
            # Calculate how many points to show up to this frame
            n_points = min(frame + 1, len(all_times))
            
            # Clear only scatter plots (PathCollection objects) from animated panels
            # Panel D (FI vs Magnitude)
            collections_to_remove = [col for col in ax_fi_mag.collections if isinstance(col, PathCollection)]
            for col in collections_to_remove:
                col.remove()
            
            # Panel E (Timeseries)
            collections_to_remove = [col for col in ax_fi_timeseries.collections if isinstance(col, PathCollection)]
            for col in collections_to_remove:
                col.remove()
            
            # Plot points for each catalog using pre-computed data
            for cat_idx in range(len(self._catalogs)):
                # Get indices for this catalog that are within n_points
                cat_indices = catalog_groups[cat_idx]
                visible_indices = cat_indices[cat_indices < n_points]
                
                if len(visible_indices) == 0:
                    continue
                
                # Get data for visible points
                times_subset = all_times[visible_indices]
                fi_subset = all_fi_values[visible_indices]
                magnitudes_subset = [all_magnitudes[i] for i in visible_indices]
                
                # Get scatter kwargs (same for all points in a catalog)
                scatter_kwargs = all_scatter_kwargs[visible_indices[0]].copy()
                
                # Handle magnitudes if available (for scatter size)
                if all(m is not None for m in magnitudes_subset):
                    scatter_sizes = [max(48.75 * mag - 43.75, 0) for mag in magnitudes_subset]
                    scatter_kwargs['s'] = scatter_sizes
                
                # Ensure facecolor is used
                if 'facecolor' not in scatter_kwargs and 'c' not in scatter_kwargs and 'color' not in scatter_kwargs:
                    scatter_kwargs['facecolor'] = 'k'
                
                # Plot on timeseries panel (E)
                ax_fi_timeseries.scatter(times_subset, fi_subset, **scatter_kwargs)
                
                # Plot on FI vs Magnitude panel (D) - only if magnitudes available
                if all(m is not None for m in magnitudes_subset):
                    ax_fi_mag.scatter(fi_subset, magnitudes_subset, **scatter_kwargs)
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                      interval=1000/fps, blit=False, repeat=False)
        
        # Determine writer based on file extension
        filename_lower = filename.lower()
        if filename_lower.endswith('.mp4'):
            try:
                writer = animation.writers['ffmpeg'](fps=fps, bitrate=1800)
            except (KeyError, RuntimeError) as e:
                raise RuntimeError(
                    "ffmpeg writer not available. Install ffmpeg to create MP4 movies.\n"
                    "You can install ffmpeg with: conda install ffmpeg\n"
                    "Or use GIF format instead (change filename to .gif)"
                ) from e
        elif filename_lower.endswith('.gif'):
            try:
                writer = animation.PillowWriter(fps=fps)
            except AttributeError:
                # Fallback for older matplotlib versions
                try:
                    writer = animation.writers['pillow'](fps=fps)
                except (KeyError, RuntimeError) as e:
                    raise RuntimeError(
                        "Pillow writer not available. Install Pillow to create GIF movies.\n"
                        "You can install Pillow with: pip install Pillow or conda install pillow"
                    ) from e
        else:
            raise ValueError("Filename must end with .mp4 or .gif")
        
        # Save animation
        print(f"  Rendering and saving movie...")
        anim.save(filename, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Movie saved successfully: {filename}")


# Example usage:
if __name__ == "__main__":

    # Create example data
    from obspy import UTCDateTime, read
    import numpy as np
    import os

    dt = UTCDateTime("2005-08-31T02:34:00")

    # ObsPy example waveform
    # st = read("https://examples.obspy.org/loc_RJOB20050831023349.z",
    #           starttime=dt, endtime=dt + 10)

    # Original Augustine data from Buurman et al. (2010)
    # Get path relative to this script: go up to package root, then into data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data')
    data_file = os.path.join(data_dir, 'Augustine_test_data_FI.mseed')
    # stub fullpath
    data_file = "/Users/jwellik/PYTHON/PKG/vdapseisutils/data/Augustine_test_data_FI.mseed"

    st = read(data_file)
    # st = st.filter("bandpass", freqmin=0.8, freqmax=25.0, corners=4)
    
    # Create Buurman_2010 instance
    buurman = Buurman_2010(
        Alower=[1, 2],
        Aupper=[10, 20],
        thresholds=[-3.0, -0.4],
        threshold_colors=["black", "red", "blue"],  # 3 colors for 2 thresholds
        analysis_window=[0.25, 5.25],
    )

    # Add stream data (stored, not plotted yet)
    buurman.add_stream(st[0], magnitude=1.0)
    buurman.add_stream(st[1], magnitude=1.3)
    buurman.add_stream(st[2], magnitude=-0.5)

    # Create a figure layout - this will plot all stored streams
    # buurman.plot_fig23()  # Combined layout (waveform + spectra + frequency index)
    buurman.plot_fig234()  # Combined layout (waveform + spectra + frequency index + timeseries)
    # Or use:
    # buurman.plot_fig2()   # Waveform only
    # buurman.plot_fig3()   # Spectra and frequency index only

    # Display the plot (optional - figures auto-display in Jupyter notebooks)
    buurman.set_spectra_xlim([0.2, 50])
    plt.show()  # Uncomment if needed in non-interactive environments
