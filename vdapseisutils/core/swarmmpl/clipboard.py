"""
Swarm like plots for matplotlib

Author: Jay Wellik
Created: 2022 June 30
Last updated: 2023 December 9


RESOURCES:
https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
"offset_formats" are the big dates on the bottom right of the axis
https://docs.obspy.org/_modules/obspy/imaging/spectrogram.html#spectrogram


TIME
----------
In order to make traces line up for waveforms and spectrograms no matter the extent of the data or gaps, both waveforms
and spectrograms are plotted against an x-axis vector of datetime objects.
For a Clipboard of n Traces, Clipboard keeps
track of:
time_extent | n-by-2 list | (minimum aboslute time, maximum absolute time) for each trace
date_extent | n-by-2 list | xlim for each trace; could be datetime objects (for 'datetime') or
                            values in seconds (for 'relative')

NOTE: tr.times() returns a vector of 0:tr.stats.sampling_rate:...
In other words, a vector of seconds from 0 for each data point. (Is this true?)

DEFAULT SETTINGS
----------
spectrogram_settings =
{
'min_frequency': 0.0, 'max_frequency': 25.0,  # 'ylim': [0.0, 25.0]
'power_range_db':[20.0, 120.0],  # How can I use this?
'window_size_s': 2.0, 'overlap': 0.86,
'log_power': True,
'cmap': 'inferno',
}

wave_settings =
{
#'min_amplitude': -1000.0, 'max_amplitude': 1000.0,  # 'ylim': [-1000.0, 1000.0]
'color':'k',
filter: {'bandpass', 'min_frequency': 1.0, 'max_frequency': 10.0, 'npoles': 4},  # include this?
}

spectra_settings =
{
'min_frequency': 0.0, 'max_frequency': 25.0,  # 'xlim': [0.0, 25.0]
'log_power': True, 'log_frequency': True,
'y_axis_range': [1.0, 5.0]
}


PROCESS:
# Load all waveforms
# Determine if sharex=True|False taxis="datetime"|"relative"
# Determine if mode="wg"|"w"|"g"|"s"
# Loop through waveforms and determine start:stops of each Trace, caluclate offset
"""

# [x] Fix tick_type="relative": Currently doesn't offset data if sharex=True
# [x] sharex=False : date_extent and time_extent need to be lists of tuples for each stream (or axis)
# [x] Slowly add back spectrogram kwargs
# [x] I don't like the way NSLC labels are added; hard to read if too many traces or small figure
# [x] Make swarmg and swarmw methods
# [x] I don't love the results from AutoDateLocator
# [x] Switch the arrangement of subplots to utilize subfigures
# [x] Eliminate waveform plot redundancy
# [x] axvline uses datetime or relative
# [x] Remove __define_t_ticks
# [x] Need more room for xaxis is sync_waves = False
# [x] sync_waves isn't working
# [x] Handle datetime xlims better for datetime and relative
# [x] Handle # of traces. What to do about single trace gappy data for spectrograms?
# TODO Allow set_tlim, set_alim, etc to be for specific Traces/figures
# [x] scroll_trace(seconds)
# TODO tick_type="relative" always starts at 0?
# TODO plot_catalog(Catalog)
# TODO align_traces(datetime)
# TODO Figure out best way to do layout (not using constrained layout?)
# TODO scroll_traces() --> Turn axis labels back on

"""
TODO Create a custom Axis object or custom Figure object for TimeSeries

stores axis_type = "datetime" | "relative"
stores xlim_time = [t_start, t_end]  # xlim as datetime objects (t[0] and t[-1] of the original data)
method to convert time to x-axis coordinates

"""

import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import Stream, UTCDateTime

from vdapseisutils.core.datasource.waveID import waveID
from vdapseisutils.style import colors as vdap_colors
from vdapseisutils.utils.timeutils import convert_timeformat
from vdapseisutils.core.maps.maps import prep_catalog_data_mpl


# Plot defaults
spectrogram_defaults = {"samp_rate": 50, "wlen": 6, "overlap": 0.5, "dbscale": True, "log_power": False,
                        "cmap": vdap_colors.inferno_u}
# spectrogram_defaults_bw = {**spectrogram_defaults, **{"cmap": "binary", "dbscale": False, "clip": [0.0, 1.0]}}
wave_defaults = {"color": "k"}

# Figure and Axis defaults
figsize_default = (10.0, 6.0)
fontsize_default = 10
w_ax_defaults = {"color": "k"}
g_ax_defaults = {"ylim": [0.1, 10.0]}
s_ax_defaults = {"color": "k", "mode": "loglog", "ylim": [1.0, 10.0]}


def _set_xaxis_relative_labels_001(ax, reference_time=None):
    """Set relative xtick labels for a xaxis with datetime labels"""

    from obspy.imaging.util import _set_xaxis_obspy_dates
    _set_xaxis_obspy_dates(ax)  # set xticks as datetimes (always); doesn't hurt to do this again

    if not reference_time:
        reference_time = ax.get_xlim()[0]  # earliest time stamp on the x-axis

    xticks = ax.get_xticks()
    ax.set_xticks(xticks[1:])  # reset the xticks so that set_xticklabels is happy
    xticklabels = [str(round(x)) for x in ((xticks[1:] - xticks[0]) * 86400 / 60)]  # minutes
    ax.set_xticklabels(xticklabels)


def _set_xaxis_relative_labels(ax, reference_time=None):
    """Set relative xtick labels for a xaxis with datetime labels"""

    from obspy.imaging.util import _set_xaxis_obspy_dates
    _set_xaxis_obspy_dates(ax)  # set xticks as datetimes (always); doesn't hurt to do this again

    if not reference_time:
        reference_time = ax.get_xlim()[0]  # earliest time stamp on the x-axis

    xticks = ax.get_xticks()
    nticks = len(xticks) - 1
    ax.set_xticks(xticks[1:])  # reset the xticks so that set_xticklabels is happy
    xticklabels = [str(round(x)) for x in ((xticks[1:] - xticks[0]) * 86400 / 60)]  # minutes
    ax.set_xticklabels(xticklabels)


def t2axiscoords(times, textent, axextent, unit="datetime"):
    """T2AXISCOORDS Converts time to axis coordinates. Requires the axis object and the actual time extent for the axis
    textent should be a list-like pair of datetime objects
    axextent should be the xlim of the axis (in sample units, in other words)
    unit : Are times given as datetimes or "relative" seconds from the beginning of the axis?
    """

    # input
    times = [UTCDateTime(t) for t in times]  # convert to UTCDateTime
    t1 = UTCDateTime(textent[0])  # convert t1 to UTCDateTime
    t2 = UTCDateTime(textent[1])  # convet t2 to UTCDateTime
    t0 = t1  # preserve the original t1

    if unit == "datetime":
        xlim = axextent  # original extent of x axis
        axis_length = xlim[1] - xlim[0]  # length of original axis
        plot_duration_min = (t2 - t1) / 60  # duration in minutes that axis is meant to represent
        axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes
        x = [xlim[0] + ((t - t0) * axis_samp_rate) for t in times]

    else:  # "relative"
        x = times

    return x


def plot_trace(tr, mode="wg", tick_type="datetime", relative_offset=0, fig=None,
               wave_settings=wave_defaults, spectrogram_settings=spectrogram_defaults):
    """PLOT_TRACE Plots a single Trace and returns a MatPlotLib Figure object"""

    # TODO Add check to verify tr is a single Trace

    if not fig:
        fig = plt.figure(figsize=figsize_default)

    if mode == "wg":
        nrows = 2
        height_ratios = [1, 3]
        waxn = 0
        gaxn = 1
    elif mode == "w":
        nrows = 1
        height_ratios = [1]
        waxn = 0
        gaxn = None
    elif mode == "g":
        nrows = 1
        height_ratios = [1]
        waxn = None
        gaxn = 0

    gs = fig.add_gridspec(nrows, 1, height_ratios=height_ratios, hspace=0.0)
    for i in range(nrows):
        fig.add_subplot(gs[i])

    if waxn is not None:
        plot_wave(tr, tick_type=tick_type, relative_offset=relative_offset, ax=fig.axes[waxn], **wave_settings)
    if gaxn is not None:
        plot_spectrogram(tr, tick_type=tick_type, relative_offset=relative_offset, ax=fig.axes[gaxn],
                         **spectrogram_settings)

    # NSLC Label - Horizontal label with two rows
    s = tr.id  # returns net.sta.loc.cha
    idx = s.index(".", 3)  # ? gets second instance of "." assuming NN.SS....
    s1, s2 = s[:idx], s[idx + 1:]
    fig.axes[-1].text(
        # -0.01, 0.67, s1 + "\n" + s2, transform=fig.axes[-1].transAxes, rotation='horizontal',
        -0.01, 0.67, idx, transform=fig.axes[-1].transAxes, rotation='horizontal',
        horizontalalignment="right", verticalalignment='center', fontsize=fontsize_default,
    )

    # fig.subplots_adjust(hspace=None)  # Remove horizontal space between subplots
    return fig


def plot_wave(tr, tick_type="datetime", relative_offset=0, color="k", ax=None, **kwargs):
    """PLOT_WAVE Plots waveform of single Trace. Returns a MatPlotLib Axes object"""

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize_default)

    # Convert time values to datetime objects
    if tick_type == "datetime":
        times_w = [(tr.stats.starttime + timedelta(seconds=t)).datetime for t in
                   tr.times()]  # time vector for the waveform (w)
    else:  # "relative"
        times_w = [relative_offset + t for t in tr.times()]
    ax.plot(times_w, tr.data, color=color, **kwargs)
    ax.yaxis.set_ticks_position("right")
    ax.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{:2.1f}"))

    return ax


def plot_spectrogram(tr, samp_rate=None, wlen=2.0, overlap=0.86, dbscale=True, log_power=False,
                     cmap=vdap_colors.inferno_u, tick_type="datetime", relative_offset=0, ax=None):
    """PLOT_WAVE Plots spectrogram of single Trace. Returns a MatPlotLib Axes object"""

    # TODO Don't make spectrogram if data are empty or if all NaN

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    from datetime import timedelta
    from obspy.imaging.spectrogram import _nearest_pow_2

    if samp_rate:
        tr.resample(float(samp_rate))
    else:
        samp_rate = tr.stats.sampling_rate

    # data and sample rates
    fs = tr.stats.sampling_rate
    signal = tr.data

    # Determine variables
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(signal)

    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    # if mult is not None:
    #     mult = int(_nearest_pow_2(mult))
    #     mult = mult * nfft
    nlap = int(nfft * float(overlap))

    signal = signal - signal.mean()

    frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=nfft, noverlap=nlap, scaling='spectrum')

    # db scale and remove zero/offset for amplitude
    if dbscale:
        Sxx = 10 * np.log10(Sxx[1:, :])
    else:
        Sxx = np.sqrt(Sxx[1:, :])
    frequencies = frequencies[1:]

    # vmin, vmax = clip
    # if vmin < 0 or vmax > 1 or vmin >= vmax:
    #     msg = "Invalid parameters for clip option."
    #     raise ValueError(msg)
    # _range = float(Sxx.max() - Sxx.min())
    # vmin = Sxx.min() + vmin * _range
    # vmax = Sxx.min() + vmax * _range
    # norm = Normalize(vmin, vmax, clip=True)

    # Convert time values to datetime objects
    start_date = tr.stats.starttime.datetime
    if tick_type == "datetime":
        times_g = [start_date + timedelta(seconds=t) for t in times]  # time vector for g_kwargs (g)
    else:  # "relative"
        times_g = [relative_offset + t for t in times]

    # Plot the g_kwargs with dates on the x-axis
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize_default)

    ax.pcolormesh(times_g, frequencies, Sxx, shading='auto', cmap=cmap)  # plot g_kwargs
    if log_power:
        ax.set_yscale('log')  # Use a logarithmic scale for the y-axis
    ax.set_ylim(0.1, samp_rate / 2.0)  # Set the frequency range to 0.5 - 25 Hz
    ax.yaxis.set_ticks_position("right")
    ax.set_xlabel("")
    ax.set_ylabel("")

    data = {"freq": frequencies, "times": times_g, "power": Sxx}

    return ax, data


class ClipboardClass(plt.Figure):

    def __init__(self, st=Stream(), mode="wg", figsize=None,
                 tick_type="datetime", sync_waves=True, force_length=True,
                 **kwargs,
                 ):
        """
        CLIPBOARD Creates subplots of waveforms or spectrograms for an ObsPy Stream object.
        X ticks can be plotted as datetimes or as "relative" seconds from the start of each Trace.
        Similarly, Traces can be synced by aboslute time or by the beginning of each Trace

        This class will not merge any Traces. Thus, it allows for different events from the station to be plotted
        alongside one another. Therefore, it is important to make sure all Streams are pre-processed before they are
        passed to ClipboardClass

        :param st: Stream object with one or more Traces to plot
        :param mode: "wg" waveform and spectrogram | "w" waveforms only | "g" spectrogram only
        :param figsize: tuple passed to matplotlib.figure
        :param g: dictionary of settings for spectrogram
            defaults: {"samp_rate": 50, "wlen": 6, "overlap": 0.5, "dbscale": True, "log_power": False, "cmap": vdap_colors.inferno_u}
        :param tick_type: "datetime" | "relative" seconds after start of Trace
        :param sync_waves: True (plots are synced by aboslute time of Trace objects) | False
        :param force_length: Forces all subplots to be the same length even if data limits are different
        :param kwargs:
        """

        # self.st = Stream(st.copy())  # Ensure object is Stream (converts Trace)
        self.st = st.copy()  # Stream object
        self.n_traces = len(st)

        # Figure object
        self.figsize = figsize if figsize else (8.5, np.max([len(self.st) * 2, 3]))

        # Plot type and Time
        self.mode = mode
        # self.sync_waves = sync_waves

        # Plot settings
        self.spectrogram_settings = spectrogram_defaults
        self.wave_settings = wave_defaults

        # Time axis information
        self.taxis = dict()
        self.taxis["tick_type"] = tick_type
        self.taxis["sync_waves"] = sync_waves
        self.taxis["force_length"] = force_length
        self.taxis["time_lim"] = np.empty((self.n_traces, 2),
                                          dtype=object)  # n-by-2 list of xlims in datetime units, regardless of tick_type
        self.taxis["xlim"] = np.empty((self.n_traces, 2),
                                      dtype=object)  # n-by-2 list of xlims in axis units, regardless of tick_type (should be what you get if you call ax.get_xlim)

        # Initialize the figure
        # super().__init__(figsize=self.figsize, layout="constrained", **kwargs)  # constrained layout
        super().__init__(figsize=self.figsize, **kwargs)  # unconstrained layout

        # Create the subfigures
        self.subfigs = self.subfigures(self.n_traces, 1)
        self.suptitle('Clipboard', fontsize=fontsize_default)

    # Plots, Annotations, Labels
    def plot(self):

        st = self.st.copy()

        i = 0
        for sf in self.subfigs:
            plot_trace(st[i], mode=self.mode, tick_type=self.taxis["tick_type"], fig=self.subfigs[i],
                       wave_settings=self.wave_settings, spectrogram_settings=self.spectrogram_settings)
            i += 1

        self._set_axes()

    def axvline(self, t, *args, color="red", unit="datetime", subfigs=None, **kwargs):
        """AXVLINE Adds a vertical line across all axes. Default color='red'"""
        ### ? Will this work for all use cases, tick_type = "datetime" | "relative" ++ sync_waves = True | False


        t = t if isinstance(t, (tuple, list, set)) else [t]  # convert to list, if necessary
        # for i in range(len(self.st)):
        #     for j in range(self.nax):
        # for sf in self.subfigs:
        subs = self.subfigs[subfigs] if subfigs else self.subfigs  # Allows users to specify which subfigures to plot
        for sf in subs:
            i = 0
            for ax in sf.axes:
                # n = i*self.nax+j
                x = t2axiscoords(t, self.taxis["time_lim"][i], ax.get_xlim(),
                                 unit=unit)  # convert times to x axis coordinates
                [ax.axvline(x_, *args, color=color, **kwargs) for x_ in x]  # axvline can not take a list
                i += 1

    # def suptitle(self, text):
    #     self.suptitle(text)

    def remove_labels(self):

        # if show_labels == False, remove all xticks and yticks
        for sf in self.subfigs:
            for ax in sf.axes:  # Remove all axes
                ax.set_xticks([])  # Must remove ticks and ticklabels
                ax.set_xticklabels([])  # bc both have already been set earlier in code
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                for text in ax.texts:
                    text.remove()

    # Plot Settings
    def set_spectrogram(self, **kwargs):
        self.spectrogram_settings = {**self.spectrogram_settings, **kwargs}

    def set_wave(self, **kwargs):
        self.wave_settings = {**self.wave_settings, **kwargs}

    # Axes settings
    def _set_axes(self):
        # Handles sync_waves, tick_type, etc.
        # Assumes that all axes have a vector of datetime axes or floats of seconds as the x values

        st = self.st.copy()

        # Determine xlim for axes and relative offsets (for relative & sharex=True)
        starttimes = [tr.stats.starttime for tr in st]  # starttime for every trace
        endtimes = [tr.stats.endtime for tr in st]  # endtime for every trace
        self.offset_sec = [tr.stats.starttime - min(starttimes) for tr in
                           st]  # offset in seconds from mininmum start to trace start (only used for tick_type=relative)

        # time_lim (define xlim in time units)
        if self.taxis["sync_waves"]:
            self.taxis["time_lim"] = [(min(starttimes).datetime, max(endtimes).datetime)] * len(st)  # maximum start:end extent across all traces
        else:
            self.taxis["time_lim"] = [(tr.stats.starttime.datetime, tr.stats.endtime.datetime) for tr in st]  # start:ends for each Trace

        # xlim (define xlim in axis data units)
        if self.taxis["sync_waves"]:
            if self.taxis["tick_type"] == "datetime":
                self.taxis["xlim"] = self.taxis["time_lim"].copy()
                xlabel = "Time"
            else:
                self.taxis["xlim"] = [(0, max(endtimes) - min(starttimes))] * len(st)  # length of maximum start:end extent in seconds
                xlabel = "Time (s)"
        else:
            if self.taxis["tick_type"] == "datetime":
                self.taxis["xlim"] = self.time_extent
                xlabel = "Time"
            else:
                self.taxis["xlim"] = [(0, tr.stats.endtime - tr.stats.starttime) for tr in st]  # length of maximum start:end extent in seconds
                self.offset_sec = [0.0] * len(self.st)  # no offsets
                xlabel = "Time (s)"

        # Adjust for force_length
        if self.taxis["force_length"]:

            # Find the maximum amount of time reprsented on x-axis
            # max_length_x = self.taxis["xlim"][0][1] - self.taxis["xlim"][0][0]  # max_length initialized as val of first axis
            # max_length_t = self.taxis["time_lim"][0][1] - self.taxis["time_lim"][0][0]  # max_length initialized as val of first axis
            # for i, xlim in enumerate(self.taxis["xlim"]):
            #     print(xlim)
            #     max_length_x = xlim[1] - xlim[0] if xlim[1] - xlim[0] > max_length_x else max_length_x
            #     max_length_t = self.taxis["time_lim"][i][1] - self.taxis["time_lim"][i][1][0] if self.taxis["time_lim"][i][1][1] - self.taxis["time_lim"][i][1][0] > max_length_t else max_length_t

            length_x = []
            length_t = []
            for xlim, tlim in zip(self.taxis["xlim"], self.taxis["time_lim"]):
                length_x.append(xlim[1] - xlim[0])
                length_t.append(tlim[1] - tlim[0])

            max_length_x = max(length_x)
            max_length_t = max(length_t)

            # Reset time_lim and xlim accordingly
            for i, xlim in enumerate(self.taxis["xlim"]):
                self.taxis["xlim"][i] = (
                self.taxis["xlim"][i][0], self.taxis["xlim"][i][0] + max_length_x)  # reset xlim
            for i, xlim in enumerate(self.taxis["time_lim"]):
                self.taxis["time_lim"][i] = (
                self.taxis["time_lim"][i][0], self.taxis["time_lim"][i][0] + max_length_t)  # reset time_lim

        # set xlim (Actually set the xlim on the plot axes)
        for i, sf in enumerate(self.subfigs):
            [ax.set_xlim(self.taxis["xlim"][i]) for ax in sf.axes]

        # Set all x axis ticks and ticklabels
        for i, ax in enumerate(self.get_axes()):
            if self.taxis["tick_type"] == "datetime":
                loc = mdates.AutoDateLocator(minticks=4, maxticks=7)  # from matplotlib import dates as mdates
                formatter = mdates.ConciseDateFormatter(loc)
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(formatter)

                import warnings
                warnings.simplefilter("ignore", UserWarning)
                xtl = ax.get_xticklabels()
                xt = ax.get_xticks()
                xtl[0] = mdates.num2date(xt[0]).strftime("%Y-%m-%d\n%H:%M")
                ax.set_xticklabels(xtl)
                warnings.simplefilter("default", UserWarning)
            else:
                # self.tick_type=="relative" automatically produces xunits in seconds for both spectrograms and waveforms
                # so no need to do anything fancy here
                pass

        # if sync_waves, put axes tick labels on top of top plot, remove middle axes
        if self.taxis["sync_waves"]:
            if len(self.get_axes()) >= 2:  # if len>=2 bc 1 st will produce 2 axes
                self.get_axes()[0] = self.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
                for i in range(1,
                               len(self.get_axes()) - 1):  # Remove middle axes (does not enter for loop if only 2 axes)
                    self.get_axes()[i].set_xticks([])  # Must remove ticks and ticklabels
                    self.get_axes()[i].set_xticklabels([])  # bc both have already been set earlier in code
        self.get_axes()[-1].set_xlabel(xlabel, fontsize=fontsize_default)  # Set x axis label on last plot

    # All axes
    def set_tlim(self, tlim):
        for sf in self.subfigs:
            [ax.set_xlim(tlim) for ax in sf.axes]

    # Waveform axes
    def set_alim(self, alim):  # Set amplitude limit
        if self.mode != "g":
            for sf in self.subfigs:
                sf.axes[0].set_ylim(alim)

    # Spectrogram axes
    def set_flim(self, flim):
        if self.mode != "w":  # Only if plots include a spectrogram
            n = 1 if self.mode == "wg" else 0  # which axis is the spectrogram
            for sf in self.subfigs:  # loop through all subfigures
                sf.axes[n].set_ylim(flim)

    # Power spectral range (corresponds to color range of spectrogram)
    def set_prange(self, prange):
        if self.mode != "w":
            n = 1 if self.mode == "wg" else 0
            for sf in self.subfigs:
                pass  # Set the min:max for the color range

    def scroll_traces(self, idx=None, seconds=None):
        """SCROLL_TRACE Scross trace forward or backward (-) by S seconds. Inputs are lists"""

        if not isinstance(idx, list) or not isinstance(seconds, list):
            raise ValueError("Trace index and Scroll seconds must be provided as lists of the same size.")
        if len(idx) != len(seconds):
            raise ValueError("Trace index and Scroll seconds must be provided as lists of the same size.")

        # multiply by -1 bc if you want to move the trace one direction, you need to move the axis the other direction
        seconds = list(np.array(seconds) * -1)

        offset_t = [timedelta(seconds=float(s)) for s in seconds]
        offset_x = seconds
        if self.taxis["tick_type"] == "datetime":
            offset_x = offset_t

        for i, x, t in zip(idx, offset_x, offset_t):
            self.taxis["xlim"][i] = (self.taxis["xlim"][i][0] + x, self.taxis["xlim"][i][1] + x)
            self.taxis["time_lim"][i] = (self.taxis["time_lim"][i][0] + t, self.taxis["time_lim"][i][1] + t)

        # set xlim (Actually set the xlim on the plot axes) -- resets all axes xlims, not just updated ones, should be fine
        for i, sf in enumerate(self.subfigs):
            [ax.set_xlim(self.taxis["xlim"][i]) for ax in sf.axes]


def Clipboard(st=Stream(), **kwargs):
    return plt.figure(st=st, FigureClass=ClipboardClass, **kwargs)


class TimeSeries(plt.Axes):
    """Creates time-series Axes, mostly for geophysical data"""

    name = "time-series"

    def __init__(self, fig=None, *args, tick_type="datetime", **kwargs):
        """
        Initializes a TimeSeries plot as an Axes object.

        Parameters:
        - fig: The figure to which this Axes belongs (created if None).
        - tick_type: Type of ticks on the x-axis (default: "datetime").
        - *args, **kwargs: Additional arguments for Axes initialization.
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 2), dpi=300)  # Default figure creation

        super().__init__(fig, *args, **kwargs)
        fig.add_axes(self)  # Ensure the Axes is part of the figure

        self.tick_type = tick_type
        self.time_lim = []  # Should always be datetime objects

        # xtick labels
        if self.tick_type == "datetime":
            # self.xaxis_date()
            # loc = mdates.AutoDateLocator()
            # formatter = mdates.ConciseDateFormatter(loc, show_offset=False)
            # self.ax.xaxis.set_major_locator(loc)
            # self.ax.xaxis.set_major_formatter(formatter)
            from obspy.imaging.util import _set_xaxis_obspy_dates
            _set_xaxis_obspy_dates(self)
        else:  # tick_type == "relative"
            pass

    def _set_xlim_auto(self):
        print("Set xlim")
        if self.time_lim:
            # Flatten the list of tuples and find the min and max
            all_values = [value for tup in self.time_lim for value in tup]
            smallest_value = min(all_values)
            largest_value = max(all_values)
            self.set_xlim(smallest_value, largest_value)
            print(all_values)


    def plot(self, t, data,  *args, units="datetime", **kwargs):
        self.scatter(convert_timeformat(t, "datetime"), data, *args, **kwargs)
        self.time_lim.append([min(convert_timeformat(t, "datetime")), max(convert_timeformat(t, "datetime"))])
        self._set_xlim_auto()

    def scatter(self, t, data, *args, units="datetime", **kwargs):
        self.scatter(convert_timeformat(t, "datetime"), data, *args, **kwargs)
        self.time_lim.append([min(convert_timeformat(t, "datetime")), max(convert_timeformat(t, "datetime"))])

    def axvline(self, t, *args, units="datetime", **kwargs):
        self.axvline(convert_timeformat(t, "datetime"), *args, **kwargs)
    def axvspan(self, t, *args, units="datetime", **kwargs):
        self.axvspan(convert_timeformat(t, "datetime"), *args, **kwargs)

    def plot_catalog(self, catalog, yaxis_type="depth", s="magnitude", c="time", alpha=0.5, **kwargs):

        catdata = prep_catalog_data_mpl(catalog, time_format="datetime")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s
        if c == "time":
            c = catdata["time"]
        else:
            c = c

        if yaxis_type == "depth":
            self.scatter(catdata["time"], catdata["depth"], s=s, c=c, alpha=alpha, **kwargs)
        if yaxis_type == "magnitude":
            self.scatter(catdata["time"], catdata["mag"], s=s, c=c, alpha=alpha, **kwargs)

    def plot_waveform(self, tr, *args, **kwargs):
        # t = tr.times("datetime")
        # data = tr.data
        # self.plot(t, data, *args, **kwargs)
        # self.set_xlim([t[0], t[-1]])
        plot_wave(tr, *args, ax=self, **kwargs)
        self.time_lim.append([min(convert_timeformat(tr.data, "datetime")), max(convert_timeformat(tr.data, "datetime"))])
        self._set_xlim_auto()

    def plot_spectrogram(self, tr, *args, **kwargs):
        plot_spectrogram(tr, *args, ax=self, **kwargs)
        self.time_lim.append([min(convert_timeformat(tr.data, "datetime")), max(convert_timeformat(tr.data, "datetime"))])
        self._set_xlim_auto()

    def imshow(self, t, img, **kwargs):
        xmin, xmax = convert_timeformat(min(t), "datetime"), convert_timeformat(max(t), "datetime")
        self.imshow(img, **kwargs)
        self.set_xlim(xmin, xmax)  # ? set xextent?








