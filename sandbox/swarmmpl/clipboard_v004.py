"""
Swarm like plots for matplotlib

Author: Jay Wellik
Created: 2022 June 30
Last updated: 2023 October 18


RESOURCES:
https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
"offset_formats" are the big dates on the bottom right of the axis

PROCESS:
# Load all waveforms
# Determine if sharex=True|False taxis="datetime"|"relative"
# Determine if mode="wg"|"w"|"g"|"s"
# Loop through waveforms and determine start:stops of each Trace, caluclate offset
"""

"""
    Aaron Wech's colormap
	colors=cm.jet(np.linspace(-1,1.2,256))
	color_map = LinearSegmentedColormap.from_list('Upper Half', colors)
"""


# [x] Remove unnecssary ticklabels
# TODO Implement sharex=False
# [x] I don't like the way NSLC labels are added; hard to read if too many traces or small figure
# TODO Not satisfied with colormap
# TODO XTICKLABELS ARE GETTING MADE CORRECTLY, BUT NOT PLACED CORRECTLY - The ones that are outside the time window are not getting removed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime

from vdapseisutils.core.datasource.nslcutils import getNSLCstr
from vdapseisutils.sandbox.swarmmpl import colors as vdap_colors


def t2axiscoords(times, textent, axextent, tick_type="datetime"):
    """T2AXISCOORDS Converts time to axis coordinates. Requires the axis object and the actual time extent for the axis
    textent should be a list-like pair of datetime objects
    axextent should be the xlim of the axis (in sample units, in other words)

    """

    # input
    times = [UTCDateTime(t) for t in times]  # convert to UTCDateTime
    t1 = UTCDateTime(textent[0])  # convert t1 to UTCDateTime
    t2 = UTCDateTime(textent[1])  # convet t2 to UTCDateTime
    t0 = t1  # preserve the original t1

    if tick_type == "datetime":
        xlim = axextent  # original extent of x axis
        axis_length = xlim[1] - xlim[0]  # length of original axis
        plot_duration_min = (t2 - t1) / 60  # duration in minutes that axis is meant to represent
        axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes

    if tick_type == "relative":
        # This is if the time axis should be in seconds instead of datetime
        raise Exception("Relative time axis not yet supported :-(")

    x = [xlim[0] + ((t - t0) * axis_samp_rate) for t in times]

    return x


def __define_t_ticks__(trange, xrange, tick_type="datetime"):
    """
    Returns appropriate locations (in original axis coordinates) and labels for an x axis that is meant to span t1 to t2

    :param t1:
    :param t2:
    :param ax:
    :param tick_type:
    :return:
    """

    # input
    t1 = UTCDateTime(trange[0])  # convert t1 to UTCDateTime
    t2 = UTCDateTime(trange[1])  # convet t2 to UTCDateTime
    tA = t1
    tB = t2

    xlim = xrange  # original extent of x axis
    axis_length = xlim[1]-xlim[0]  # length of original axis
    plot_duration_min = (t2-t1)/60  # duration in minutes that axis is meant to represent
    axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes
    # plot_duration_samp = 2400  # axis_length --> ax2.get_xlim()

    # determine axis format based on length of time represented
    # there's gotta be a better way to do this
    if plot_duration_min <= 2:  # 2 minutes or less
        nticks = 6
        tick_format_0 = "%Y/%m/%d %H:%M:%S"
        tick_format = '%H:%M:%S'
        t1 = pd.Timestamp(t1.datetime).round("10s")  # round to nearest 10s
        t2 = pd.Timestamp(t2.datetime).round("10s")
    elif plot_duration_min <= 10:  # 2 minutes to 10 minutes
        nticks = 6
        tick_format_0 = "%Y/%m/%d %H:%M"
        tick_format = "%H:%M"
        t1 = pd.Timestamp(t1.datetime).round("2T")  # round to nearest 10 minutes
        t2 = pd.Timestamp(t2.datetime).round("2T")
    elif plot_duration_min <= 60:  # 10 minutes to 1 hour
        nticks = 6
        tick_format_0 = "%Y/%m/%d %H:%M"
        tick_format = "%H:%M"
        t1 = pd.Timestamp(t1.datetime).round("2T")  # round to nearest 10 minutes
        t2 = pd.Timestamp(t2.datetime).round("2T")
    elif plot_duration_min <= 60*24:  # 1 hour to 1 day
        nticks = 7
        tick_format_0 = "%Y/%m/%d %H:00"
        tick_format = "%H"
        t1 = pd.Timestamp(t1.datetime).round("1H")  # round to nearest 1 hr
        t2 = pd.Timestamp(t2.datetime).round("1H")
    elif plot_duration_min <= 60*24*7:  # 1 day to 1 week
        nticks = 7
        tick_format_0 = "%Y/%m/%d"
        tick_format = "%m/%d"
        t1 = pd.Timestamp(t1.datetime).round("1D")  # round to nearest 1 day
        t2 = pd.Timestamp(t2.datetime).round("1D")
    else:  # greater than 1 week
        nticks = 7
        tick_format_0 = "%Y/%m/%d"
        tick_format = "%m/%d"
        t1 = pd.Timestamp(t1.datetime).round("1D")  # round to nearest 1 day
        t2 = pd.Timestamp(t2.datetime).round("1D")

    dr = pd.date_range(t1, t2, periods=nticks)
    ticks = np.linspace(xrange[0], xrange[1], nticks)

    if tick_type == "days":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate / 60 / 60 / 24
        axis_label = "Days"
    elif tick_type == "hours":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate / 60 / 60
        axis_label = "Hours"
    elif tick_type == "minutes":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate / 60
        axis_label = "Minutes"
    elif tick_type == "relative" or tick_type == "seconds":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate
        axis_label = "Seconds"
    else:  #  tick_type == "datetime":
        dr = dr[(dr >= pd.Timestamp(tA.datetime)) & (dr <= pd.Timestamp(tB.datetime))]
        ticks = t2axiscoords(dr, trange, xrange, tick_type="datetime")
        tick_labels = [d.strftime(tick_format) for d in dr]
        tick_labels[0] = dr[0].strftime(tick_format_0)
        axis_label = "Time"

    return ticks, tick_labels, axis_label


class Clipboard(plt.Figure):

    def __init__(self, st=Stream(), mode="wg",
                 figsize=None,
                 spectrogram={}, w_ax={}, g_ax={}, s_ax={},
                 tick_type="datetime",
                 sharex=True,
                 **kwargs,
                 ):

        # Defaults
        spectrogram_defaults = {"log": False, "samp_rate": 50, "dbscale": True, "per_lap": 0.5, "mult": 25.0, "wlen": 6,
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
        self.sharex = sharex

        # Figure object
        self.figsize = figsize if figsize else (8.5, np.max([len(self.st)*2, 3]))

        # Other settings
        self.spectrogram = {**spectrogram_defaults, **spectrogram}
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

        # Determine offsets, reftime, etc.
        starttimes = [tr.stats.starttime for tr in st]  # starttime for every trace
        endtimes = [tr.stats.endtime for tr in st]  # endtime for every trace
        self.data_extent = [min(starttimes), max(endtimes)]  # maximum start:end extent across all traces
        extent_len = (self.data_extent[1] - self.data_extent[0]) * st[
            0].stats.sampling_rate  # length of maximum extent in samples
        offset_t = [tr.stats.starttime - self.data_extent[0] for tr in st]  # offset of each trace in seconds
        print(self.data_extent)
        print(extent_len)

        # Plot the data
        for i, tr in enumerate(st):
            axn = i * self.nplots

            # Define offset and xlim for this trace (assume waveform for now)
            offset = tr.stats.starttime - self.data_extent[0]
            xlim = [0 - (offset * tr.stats.sampling_rate), extent_len - (
                    offset * tr.stats.sampling_rate)]  # necessary xlim in samples for trace to align with other traces

            # Plot the secondary/top axis for waveforms if "wg"
            if self.mode == "wg":
                self.add_subplot(gs[axn])
                self.axes[axn].plot(tr.data, **self.w_ax)  # Just plot points, not time
                self.axes[axn].set_xlim(xlim)
                ticks, labels, xlabel = __define_t_ticks__(self.data_extent, self.axes[axn].get_xlim(), tick_type=self.tick_type)
                self.axes[axn].set_xticks(ticks)
                self.axes[axn].set_xticklabels(labels, fontsize=10)
                self.axes[axn].yaxis.set_ticks_position('right')
                print(xlim)

            # Plot the primary/bottom axis
            if self.mode == "wg" or self.mode == "g":
                axn = axn+1 if self.mode == "wg" else axn  # increment axn by 1 if this is the second plot per trace
                xlim = [xlim[0] / self.spectrogram["samp_rate"], xlim[1] / self.spectrogram["samp_rate"]]  # adjust xlim for g_kwargs
                self.add_subplot(gs[axn])
                tr.g_kwargs(axes=self.axes[axn], **self.spectrogram)
                self.axes[axn].set_xlim(xlim)
                ticks, labels, xlabel = __define_t_ticks__(self.data_extent, self.axes[axn].get_xlim(), tick_type=self.tick_type)
                self.axes[axn].set_xticks(ticks)
                self.axes[axn].set_xticklabels(labels, fontsize=10)
                self.axes[axn].yaxis.set_ticks_position('right')
                print(xlim)
            elif self.mode == "w":
                offset = tr.stats.starttime - self.data_extent[0]
                self.add_subplot(gs[axn])
                self.axes[axn].plot(tr.data, **self.w_ax)  # Just plot points, not time
                self.axes[axn].set_xlim(xlim)
                ticks, labels, xlabel = __define_t_ticks__(self.data_extent, self.axes[axn].get_xlim(), tick_type=self.tick_type)
                self.axes[axn].set_xticks(ticks)
                self.axes[axn].set_xticklabels(labels, fontsize=10)
                self.axes[axn].yaxis.set_ticks_position('right')
                print(xlim)

            # NSLC Label - Horizontal label with two rows
            s = getNSLCstr(tr)
            idx = s.index(".",3)  # ? gets second instance of "." assuming NN.SS....
            s1, s2 = s[:idx], s[idx + 1:]
            self.axes[axn].text(
                -0.01, 0.67, s1 + "\n" + s2, transform=self.axes[axn].transAxes, rotation='horizontal',
                horizontalalignment="right", verticalalignment='center', fontsize=10,
            )

            # NSLC Label - Vertical label w 1 row
            # self.subplots[axn].text(
            #     -0.01, 0.5, getNSLCstr(tr), transform=self.subplots[axn].transAxes, rotation='vertical',
            #     horizontalalignment='right', verticalalignment='center', fontsize=12,
            # )


        if self.sharex:
            # Put axes on top of top plot
            # Remove middle axes
            if len(self.get_axes()) >= 2:  # if len>2 bc 1 st will produce two axes
                self.get_axes()[0] = self.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
                for i in range(1, len(self.get_axes())-1):  # Remove middle axes (does not enter for loop if only 2 axes)
                    self.get_axes()[i] = self.get_axes()[i].set_xticks([])
        self.axes[-1].set_xlabel(xlabel, fontsize=10)  # Set x axis label on last plot

    def axvline(self, t, *args, color="red", **kwargs):
        """AXVLINE Adds a vertical line across all axes. Default color='red'"""
        t = t if isinstance(t, (tuple, list, set)) else [t]  # convert to list, if necessary
        for i, ax in enumerate(self.axes):
            x = t2axiscoords(t, self.data_extent, ax.get_xlim(), tick_type=self.tick_type)  # convert times to x axis coordinates
            [ax.axvline(x_, *args, color=color, **kwargs) for x_ in x]  # axvline can not take a list
