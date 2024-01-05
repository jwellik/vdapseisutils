# I THINK THIS ONE CAN BE DELETED

"""
Swarm like plots for matplotlib

Author: Jay Wellik
Created: 2022 June 30
Last updated: 2023 October 31


RESOURCES:
https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
"offset_formats" are the big dates on the bottom right of the axis
https://docs.obspy.org/_modules/obspy/imaging/spectrogram.html#spectrogram


TIME
----------
In order to make traces line up for waveforms and spectrograms no matter the extent of the data or gaps, both waveforms
and spectrograms are plotted against an x-axis vector of datetime objects. For a Clipboard of n Traces, Clipboard keeps
track of:
time_extent | n-by-2 list | (minimum aboslute time, maximum absolute time) for each trace
date_extent | n-by-2 list | xlim for each trace; could be datetime objects (for 'datetime') or values in seconds (for 'relative')

NOTE: tr.times() returns a vector of 0:tr.stats.sampling_rate:... In other words, a vector of seconds from 0: for each data point.


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
# TODO Make swarmg and swarmw methods
# TODO I don't love the results from AutoDateLocator
# TODO Switch the arrangement of subplots to utilize subfigures
# [x] Eliminate waveform plot redundancy
# [x] axvline uses datetime or relative
# [x] Remove __define_t_ticks
# TODO Need more room for xaxis is sharex = False


import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import Stream, UTCDateTime

from vdapseisutils.core.datasource.nslcutils import getNSLCstr
from vdapseisutils.sandbox.swarmmpl import colors as vdap_colors
from vdapseisutils.sandbox.swarmmpl.spectrogram import swarmg


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


def plot_wave(tr, tick_type="datetime", relative_offset=0, ax=None, **kwargs):

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))

    # Convert time values to datetime objects
    if tick_type == "datetime":
        times_w = [(tr.stats.starttime + timedelta(seconds=t)).datetime for t in tr.times()]  # time vector for the waveform (w)
    else:  # "relative"
        times_w = [relative_offset + t for t in tr.times()]
    ax.plot(times_w, tr.data, **kwargs)

    return ax


def plot_spectrogram(tr, samp_rate=None, wlen=6.0, overlap=0.5, dbscale=True, log_power=False,
                  cmap=vdap_colors.inferno_u, tick_type="datetime", relative_offset=0, ax=None,
                     specgram_kwargs=None, pcolormesh_kwargs=None):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    from datetime import timedelta
    import matplotlib.dates as mdates
    from obspy.imaging.spectrogram import _nearest_pow_2

    if samp_rate:
        tr.resample(float(samp_rate))
    else:
        samp_rate = tr.stats.sampling_rate

    # data and sample rates
    fs = tr.stats.sampling_rate
    signal = tr.data

    # Define the start date and time
    start_date = tr.stats.starttime.datetime

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
    if tick_type == "datetime":
        times_g = [start_date + timedelta(seconds=t) for t in times]  # time vector for g_kwargs (g)
    else:  # "relative"
        times_g = [relative_offset + t for t in times]

    # Plot the g_kwargs with dates on the x-axis
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))

    ax.pcolormesh(times_g, frequencies, Sxx, shading='auto', cmap=cmap)  # plot g_kwargs
    if log_power:
        ax.set_yscale('log')  # Use a logarithmic scale for the y-axis
    ax.set_ylim(0.1, samp_rate/2.0)  # Set the frequency range to 0.5 - 25 Hz

    data = {"freq": frequencies, "times": times_g, "power": Sxx}

    return ax, data


def plot_trace(tr, mode="wg", tick_type="datetime", relative_offset=0, fig=None,
               specgram={'samp_rate': None, 'wlen': 6.0, 'overlap': 0.5, 'dbsacle': True, 'log_power': False, 'kwargs': None},
               colormesh={'cmap': vdap_colors.inferno_u, 'kwargs': None},
               ):

    if not fig:
        nrows = 2 if mode == "wg" else 1
        height_ratios = [1, 3] if mode == "wg" else [1]
        fig = plt.figure(figsize=(10.0, 6.0))
        gs = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios)

    if mode == "wg":
        wax = 0
        gax = 1
    elif mode == "w":
        wax = 0
        gax = None
    elif mode == "g":
        wax = None
        gax = 0
    elif mode == "s":  # Not yet implemented
        wax = None
        gax = None

    if wax:
        fig.add_subplot(gs[wax])
        fig.axes[wax] = plot_wave(tr, tick_type=tick_type, relative_offset=relative_offset, ax=fig.axes[wax])  # ? proper call to axes
    if gax:
        fig.add_subplot(gs[gax])
        fig.axes[gax] = plot_spectrogram(tr, tick_type=tick_type, relative_offset=relative_offset, ax=fig.axes[gax])  # ? proper call to axes

    # Remove space inbetween axes
    # Auto-date Locator
    # NSLC Labels

    return fig


class ClipboardClass(plt.Figure):

    def __init__(self, st=Stream(), mode="wg",
                 figsize=None,
                 g={}, w_ax={}, g_ax={}, s_ax={},
                 tick_type="datetime",
                 sharex=True,
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
        :param w_ax: dictionary of settings for waveform axes
        :param g_ax: dictionary of settings for spectrogram axes
        :param s_ax: dictionary of settings for spectra axes
        :param tick_type: "datetime" | "relative" seconds after start of Trace
        :param sharex: True (plots are synced by aboslute time of Trace objects) | False
        :param kwargs:
        """

        # Defaults
        spectrogram_defaults = {"samp_rate": 50, "wlen": 6, "overlap": 0.5, "dbscale": True, "log_power": False, "cmap": vdap_colors.inferno_u}
        # spectrogram_defaults_bw = {**spectrogram_defaults, **{"cmap": "binary", "dbscale": False, "clip": [0.0, 1.0]}}
        w_ax_defaults = {"color": "k"}
        g_ax_defaults = {"ylim": [0.1, 10.0]}
        s_ax_defaults = {"color": "k", "mode": "loglog", "ylim": [1.0, 10.0]}

        # self.st = Stream(st.copy())  # Ensure object is Stream (converts Trace)
        self.st = st.copy()  # Stream object
        self.ntr = 0  # total number of traces in self.st  # Used?
        self.n_subplots = 0  # total number of subplots  # ? Used?
        self.mode = mode  # "w", "g", "wg"
        self.nax = 2 if mode == "wg" else 1
        self.sharex = sharex

        # Figure object
        self.figsize = figsize if figsize else (8.5, np.max([len(self.st)*2, 3]))

        # Other settings
        self.g_kwargs = {**spectrogram_defaults, **g}
        self.tick_type = tick_type
        if self.mode == "wg":
            self.wg_ratio = [1, 3]  # height ratio of waveform axis : specgram axis
            self.nplots = 2  # nplots per trace
        else:
            self.wg_ratio = [1]
            self.nplots = 1

        # Axis settings (y_lim, color, etc.)
        self.g_ax = {**g_ax_defaults, **g_ax}
        self.w_ax = {**w_ax_defaults, **w_ax}
        self.s_ax = {**s_ax_defaults, **s_ax}

        super().__init__(figsize=self.figsize, **kwargs)

        self._plot_clipboard()  # cretes gridspec, all the axes, and plots the data


    def _plot_clipboard(self):
        ################################################################################################################
        # Make figure: I could put it in another method, but it's a lot of unnecessary parameter passing (for now)

        st = self.st.copy()

        # Create Figure with Subfigures - 1 per Trace
        # [old] gs = self.add_gridspec(nrows=len(self.st) * self.nplots, ncols=1, height_ratios=self.wg_ratio * len(self.st))

        # Determine xlim for axes and relative offsets (for relative & sharex=True)
        starttimes = [tr.stats.starttime for tr in st]  # starttime for every trace
        endtimes = [tr.stats.endtime for tr in st]  # endtime for every trace
        self.offset_sec = [tr.stats.starttime - min(starttimes) for tr in st]  # offset in seconds from mininmum start to trace start (only used for tick_type=relative)

        # time extent
        # The absolute time range for each axis.
        # sharex==True --> all axes are set to the min:max of all Streams
        # sharex==False --> each axis is set to min:max of its own Stream
        if self.sharex:
            self.time_extent = [(min(starttimes).datetime, max(endtimes).datetime)] * len(st)  # maximum start:end extent across all traces
        else:
            self.time_extent = [(tr.stats.starttime.datetime, tr.stats.endtime.datetime) for tr in st]  # start:ends for each Trace

        # date_extent
        # The xlim for each axis
        # sharex==True
        #    datetime==True --> same as time_extent
        #    datetime==False --> all axes are set to [0, length of time_extent]
        # sharex==False
        #    datetime==True --> same as time_extent
        #    datetime==False --> each axis is set to [0, length of its own Stream]
        if self.sharex:
            if self.tick_type == "datetime":
                self.data_extent = self.time_extent
                xlabel = "Time"
            else:
                self.data_extent = [(0, max(endtimes)-min(starttimes))] * len(st)  # length of maximum start:end extent in seconds
                xlabel = "Time (s)"
        else:
            if self.tick_type == "datetime":
                self.data_extent = self.time_extent
                xlabel = "Time"
            else:
                self.data_extent = [(0, tr.stats.endtime-tr.stats.starttime) for tr in st]  # length of maximum start:end extent in seconds
                self.offset_sec = [0.0] * len(self.st)  # no offsets
                xlabel = "Time (s)"

        # Plot the data
        for i, tr in enumerate(st):
            # plot_trace(tr, mode=self.mode, tick_type=self.tick_type, fig=<next_figure>)
            pass


        # if sharex, Put axes on top of top plot, Remove middle axes
        if self.sharex:
            # Change get_axes() lines to loop over subfigures
            if len(self.get_axes()) >= 2:  # if len>=2 bc 1 st will produce 2 axes
                self.get_axes()[0] = self.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
                for i in range(1, len(self.get_axes())-1):  # Remove middle axes (does not enter for loop if only 2 axes)
                    self.get_axes()[i].set_xticks([])       # Must remove ticks and ticklabels
                    self.get_axes()[i].set_xticklabels([])  # bc both have already been set earlier in code
        self.get_axes()[-1].set_xlabel(xlabel, fontsize=10)  # Set x axis label on last plot

    def axvline(self, t, *args, color="red", unit="datetime", **kwargs):
        """AXVLINE Adds a vertical line across all axes. Default color='red'"""
        ### ? Will this work for all use cases, tick_type = "datetime" | "relative" ++ sharex = True | False

        t = t if isinstance(t, (tuple, list, set)) else [t]  # convert to list, if necessary
        # for i, ax in enumerate(self.axes):
        for i in range(len(self.st)):
            for j in range(self.nax):
                n = i*self.nax+j
                x = t2axiscoords(t, self.time_extent[i], self.get_axes()[n].get_xlim(), unit=unit)  # convert times to x axis coordinates
                [self.get_axes()[n].axvline(x_, *args, color=color, **kwargs) for x_ in x]  # axvline can not take a list

    def remove_labels(self):
        # if I get this to work, remove it from the _plot_clipboard routine

        # if show_labels == False, remove all xticks and yticks
        for i in range(0, len(self.get_axes())):  # Remove all axes
            self.get_axes()[i].set_xticks([])  # Must remove ticks and ticklabels
            self.get_axes()[i].set_xticklabels([])  # bc both have already been set earlier in code
            self.get_axes()[i].set_yticks([])
            self.get_axes()[i].set_xlabel("")
            self.get_axes()[i].set_ylabel("")
            for text in self.get_axes()[i].texts:
                text.remove()

