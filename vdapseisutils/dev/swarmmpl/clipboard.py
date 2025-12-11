import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.projections import register_projection
from datetime import datetime, timedelta
import pandas as pd

from vdapseisutils.utils.timeutils import convert_timeformat


class TimeSeries(plt.Axes):
    """Creates time-series Axes, specifically designed for geophysical data"""

    name = "time-series"

    def __init__(self, fig, rect, tick_type="datetime", **kwargs):
        """
        Initializes a TimeSeries plot as an Axes object.

        Parameters:
        - fig: The figure to which this Axes belongs
        - rect: The position rect [left, bottom, width, height]
        - tick_type: Type of ticks on the x-axis (default: "datetime")
        - **kwargs: Additional arguments for Axes initialization
        """
        super().__init__(fig, rect, **kwargs)
        self.tick_type = tick_type
        self.time_lim = []

        # Configure x-axis for datetime display
        if self.tick_type == "datetime":
            self.xaxis_date()
            loc = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(loc, show_offset=False)
            self.xaxis.set_major_locator(loc)
            self.xaxis.set_major_formatter(formatter)

    # Override standard plotting methods to handle datetime conversion

    def plot(self, t, data, *args, units="datetime", **kwargs):
        """Plot time series data with automatic datetime conversion."""
        t_converted = convert_timeformat(t, "datetime")
        return super().plot(t_converted, data, *args, **kwargs)

    def scatter(self, t, data, *args, units="datetime", **kwargs):
        """Scatter plot with automatic datetime conversion."""
        t_converted = convert_timeformat(t, "datetime")
        return super().scatter(t_converted, data, *args, **kwargs)

    def axvline(self, t, *args, units="datetime", **kwargs):
        """Add vertical line at specified time."""
        t_converted = convert_timeformat(t, "datetime")
        return super().axvline(t_converted, *args, **kwargs)

    def axvspan(self, tmin, tmax, *args, units="datetime", **kwargs):
        """Add vertical span between specified times."""
        tmin_converted = convert_timeformat(tmin, "datetime")
        tmax_converted = convert_timeformat(tmax, "datetime")
        return super().axvspan(tmin_converted, tmax_converted, *args, **kwargs)

    def plot_catalog(self, catalog, yaxis_type="depth", s="magnitude", c="time", alpha=0.5, **kwargs):
        """
        Plot earthquake catalog data.

        Parameters:
        - catalog: Earthquake catalog data
        - yaxis_type: Y-axis data type ("depth" or "magnitude")
        - s: Size of markers (default: scale by "magnitude")
        - c: Color of markers (default: by "time")
        - alpha: Transparency of markers
        - **kwargs: Additional arguments for scatter()
        """
        catdata = prep_catalog_data_mpl(catalog, time_format="datetime")

        # Handle size parameter
        sizes = catdata["size"] if s == "magnitude" else s

        # Handle color parameter
        colors = catdata["time"] if c == "time" else c

        # Plot based on yaxis_type
        y_data = catdata["depth"] if yaxis_type == "depth" else catdata["mag"]

        return self.scatter(catdata["time"], y_data, s=sizes, c=colors, alpha=alpha, **kwargs)

    def plot_waveform(self, tr, *args, **kwargs):
        """
        Plot seismic waveform data.

        Parameters:
        - tr: Trace object containing waveform data
        - *args, **kwargs: Additional arguments for plot()
        """
        t = tr.times("datetime")
        data = tr.data
        self.plot(t, data, *args, **kwargs)
        self.set_xlim([t[0], t[-1]])
        return self

    def plot_spectrogram(self, tr, **kwargs):
        """
        Plot spectrogram of seismic data.

        Parameters:
        - tr: Trace object containing waveform data
        - **kwargs: Additional arguments for imshow()
        """
        from scipy import signal

        # Generate spectrogram
        fs = tr.stats.sampling_rate
        f, t, Sxx = signal.spectrogram(tr.data, fs)

        # Convert trace start time to datetime
        t0 = tr.stats.starttime.datetime

        # Convert relative times to absolute datetimes
        t_abs = [t0 + pd.Timedelta(seconds=dt) for dt in t]

        # Plot spectrogram
        im = self.imshow(Sxx, extent=[t_abs[0], t_abs[-1], f[0], f[-1]],
                         aspect='auto', origin='lower', **kwargs)

        return im

    def imshow_timeseries(self, data, times, **kwargs):
        """
        Display an image with proper time axis labeling.

        Parameters:
        - data: 2D array of data to display
        - times: Array of time values for the x-axis
        - **kwargs: Additional arguments for imshow()
        """
        t_converted = convert_timeformat(times, "datetime")
        t_min, t_max = min(t_converted), max(t_converted)

        # Get current extent or use defaults
        extent = kwargs.pop('extent', None)
        if extent is None:
            height, width = data.shape
            extent = [t_min, t_max, 0, height]
        else:
            extent[0], extent[1] = t_min, t_max

        return self.imshow(data, extent=extent, aspect='auto', **kwargs)



# Generate some fake seismic and earthquake data
def generate_fake_seismic_data(start_time, duration_days=10, sampling_rate=10):
    """Generate fake seismic waveform data"""
    # Create time array
    times = [start_time + timedelta(minutes=i / sampling_rate)
             for i in range(int(duration_days * 24 * 60 * sampling_rate))]

    # Generate synthetic seismic data (combination of sine waves + noise)
    n_samples = len(times)
    t_numeric = np.linspace(0, duration_days, n_samples)

    # Base signal (combination of sine waves with different frequencies)
    base_signal = (
            np.sin(2 * np.pi * 0.5 * t_numeric) +
            0.5 * np.sin(2 * np.pi * 1.5 * t_numeric) +
            0.3 * np.sin(2 * np.pi * 3.7 * t_numeric)
    )

    # Add some random spikes (earthquakes)
    n_spikes = int(duration_days * 3)  # ~3 events per day
    spike_indices = np.random.choice(range(n_samples), n_spikes, replace=False)

    data = base_signal.copy()
    for idx in spike_indices:
        # Generate a spike with exponential decay
        if idx < n_samples - 100:  # Ensure we have room for decay
            spike = np.random.uniform(5, 15)  # Random amplitude
            decay_length = np.random.randint(20, 100)
            decay = spike * np.exp(-np.linspace(0, 5, decay_length))
            data[idx:idx + decay_length] += decay

    # Add some noise
    noise_level = 0.2
    noise = np.random.normal(0, noise_level, n_samples)
    data += noise

    return times, data, spike_indices


def generate_fake_earthquake_catalog(start_time, duration_days=10, n_events=None):
    """Generate fake earthquake catalog data"""
    if n_events is None:
        n_events = int(duration_days * 3)  # ~3 events per day

    # Generate random times within the period
    event_times = [start_time + timedelta(days=np.random.uniform(0, duration_days))
                   for _ in range(n_events)]
    event_times.sort()

    # Generate random magnitudes (mostly small with a few larger ones)
    magnitudes = np.random.exponential(1.0, n_events) + 1.0

    # Generate random depths (typically between 5-30km)
    depths = np.random.gamma(2, 5, n_events) + 5

    # Generate random locations (not used in this example but included for completeness)
    latitudes = np.random.uniform(34.0, 36.0, n_events)
    longitudes = np.random.uniform(-118.5, -116.5, n_events)  # Southern California-ish

    return {
        'time': event_times,
        'magnitude': magnitudes,
        'depth': depths,
        'latitude': latitudes,
        'longitude': longitudes
    }


# Create a figure with two TimeSeries subplots
def create_dual_timeseries_example():

    # Register the custom projection
    register_projection(TimeSeries)

    # Start date for our fake data
    start_date = datetime(2023, 3, 1)
    duration_days = 7

    # Generate fake seismic and earthquake data
    times, waveform_data, spike_indices = generate_fake_seismic_data(start_date, duration_days)
    earthquake_catalog = generate_fake_earthquake_catalog(start_date, duration_days)

    # Create figure with two TimeSeries axes
    fig = plt.figure(figsize=(12, 8))

    # First TimeSeries for waveform data
    ax1 = fig.add_subplot(211, projection='time-series')
    ax1.plot(times, waveform_data, 'k-', linewidth=0.8)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Simulated Seismic Waveform')

    # Highlight some "events" with vertical spans
    for i, spike_idx in enumerate(spike_indices[:5]):  # Highlight first 5 spikes
        event_time = times[spike_idx]
        event_end = event_time + timedelta(hours=np.random.uniform(0.5, 2.0))
        ax1.axvspan(event_time, event_end, color='red', alpha=0.2)

    # Second TimeSeries for earthquake catalog
    ax2 = fig.add_subplot(212, projection='time-series')

    # Scale marker sizes by magnitude
    marker_sizes = 10 * np.exp(earthquake_catalog['magnitude'])

    # Color markers by depth
    scatter = ax2.scatter(
        earthquake_catalog['time'],
        earthquake_catalog['magnitude'],
        s=marker_sizes,
        c=earthquake_catalog['depth'],
        cmap='viridis',
        alpha=0.7
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Depth (km)')

    ax2.set_ylabel('Magnitude')
    ax2.set_ylim(bottom=0)
    ax2.set_title('Simulated Earthquake Catalog')

    # Add reference lines for significant magnitude thresholds
    ax2.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=7.0, color='red', linestyle='--', alpha=0.7)

    # Add a text label for a major event
    max_mag_idx = np.argmax(earthquake_catalog['magnitude'])
    max_mag_time = earthquake_catalog['time'][max_mag_idx]
    max_mag = earthquake_catalog['magnitude'][max_mag_idx]
    ax2.annotate(f'M{max_mag:.1f}',
                 xy=(max_mag_time, max_mag),
                 xytext=(10, 10),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round', fc='yellow', alpha=0.7))

    # Adjust layout and add a common title
    plt.tight_layout()
    fig.suptitle('Seismic Activity Analysis', fontsize=16, y=1.02)

    return fig, ax1, ax2


def main():

    # Register the custom projection
    register_projection(TimeSeries)

    # Execute the example
    fig, ax1, ax2 = create_dual_timeseries_example()
    plt.show()


if __name__ == "__main__":
    main()
