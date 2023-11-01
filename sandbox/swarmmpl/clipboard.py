"""
Swarm like plots for matplotlib

Author: Jay Wellik
Created: 2022 June 30
Last updated: 2023 October 31


RESOURCES:
https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
"offset_formats" are the big dates on the bottom right of the axis
https://docs.obspy.org/_modules/obspy/imaging/spectrogram.html#spectrogram

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


class Clipboard(plt.Figure):

    def __init__(self, st=Stream(), mode="wg",
                 figsize=None,
                 g={}, w_ax={}, g_ax={}, s_ax={},
                 tick_type="datetime",
                 sharex=True,
                 **kwargs,
                 ):

        # Defaults
        spectrogram_defaults = {"log_power": False, "samp_rate": 50, "dbscale": True, "overlap": 0.5, "wlen": 6,
                                "cmap": vdap_colors.inferno_u}
        spectrogram_defaults_bw = {**spectrogram_defaults, **{"cmap": "binary", "dbscale": False, "clip": [0.0, 1.0]}}
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
        gs = self.add_gridspec(nrows=len(self.st) * self.nplots, ncols=1, height_ratios=self.wg_ratio * len(self.st))

        # Determine xlim for axes and relative offsets (for relative & sharex=True)
        starttimes = [tr.stats.starttime for tr in st]  # starttime for every trace
        endtimes = [tr.stats.endtime for tr in st]  # endtime for every trace
        self.offset_sec = [tr.stats.starttime - min(starttimes) for tr in st]  # offset in seconds from mininmum start to trace start (only used for tick_type=relative)

        # time extent
        if self.sharex:
            self.time_extent = [(min(starttimes).datetime, max(endtimes).datetime)]  * len(st)  # maximum start:end extent across all traces
        else:
            self.time_extent = [(tr.stats.starttime.datetime, tr.stats.endtime.datetime) for tr in st]  # start:ends for each Trace

        # date_extent
        if self.sharex:
            if self.tick_type == "datetime":
                self.data_extent = self.time_extent
                xlabel = "Time"
            else:
                self.data_extent = [(0, max(endtimes)-min(starttimes))]  * len(st)  # length of maximum start:end extent in seconds
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

            # Assign indices for each axis for this trace
            if self.mode == "wg":
                waxn = i * self.nplots
                gaxn = waxn+1
                laxn = gaxn
            elif self.mode == "g":
                waxn = -1
                gaxn = i * self.nplots
                laxn = gaxn
            elif self.mode == "w":
                waxn = i * self.nplots
                gaxn = -1
                laxn = waxn

            start_date = tr.stats.starttime

            # Convert time values to datetime objects
            if self.tick_type == "datetime":
                times_w = [(start_date + timedelta(seconds=t)).datetime for t in tr.times()]  # time vector for the waveform (w)
            else:  # "relative"
                times_w = [self.offset_sec[i] + t for t in tr.times()]

            # Waveform
            if waxn > -1:
                axn = waxn
                self.add_subplot(gs[axn])
                self.axes[axn].plot(times_w, tr.data, **self.w_ax)  # Just plot points, not time
                self.axes[axn].yaxis.set_ticks_position("right")
                self.axes[axn].set_xlim(self.data_extent[i])

            # Spectrogram
            if gaxn > -1:
                axn = gaxn
                # axn = axn+1 if self.mode == "wg" else axn  # increment axn by 1 if this is the second plot per trace
                self.add_subplot(gs[axn])
                self.axes[axn] = swarmg(tr, tick_type=self.tick_type, relative_offset=self.offset_sec[i], ax=self.axes[axn], **self.g_kwargs)
                self.axes[axn].set_xlabel("")
                self.axes[axn].set_ylabel("")
                self.axes[axn].yaxis.set_ticks_position("right")
                self.axes[axn].set_xlim(self.data_extent[i])

            # NSLC Label - Horizontal label with two rows
            s = getNSLCstr(tr)
            idx = s.index(".",3)  # ? gets second instance of "." assuming NN.SS....
            s1, s2 = s[:idx], s[idx + 1:]
            self.axes[laxn].text(
                -0.01, 0.67, s1 + "\n" + s2, transform=self.axes[laxn].transAxes, rotation='horizontal',
                horizontalalignment="right", verticalalignment='center', fontsize=10,
            )

            # NSLC Label - Vertical label w 1 row
            # self.subplots[axn].text(
            #     -0.01, 0.5, getNSLCstr(tr), transform=self.subplots[axn].transAxes, rotation='vertical',
            #     horizontalalignment='right', verticalalignment='center', fontsize=12,
            # )

        # Set all x axis ticks and ticklabels
        for i, ax in enumerate(self.get_axes()):
            if self.tick_type == "datetime":
                loc = mdates.AutoDateLocator(minticks=4, maxticks=7)  # from matplotlib import dates as mdates
                formatter = mdates.ConciseDateFormatter(loc)
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(formatter)
            else:
                # self.tick_type=="relative" automatically produces xunits in seconds for both spectrograms and waveforms
                # so no need to do anything fancy here
                pass

        # if sharex, Put axes on top of top plot, Remove middle axes
        if self.sharex:
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
