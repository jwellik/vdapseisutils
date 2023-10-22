"""
Swarm like plots for matplotlib

Author: Jay Wellik
Created: 2022 June 30
Last updated: 2023 October 18


RESOURCES:
https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
"offset_formats" are the big dates on the bottom right of the axis
"""

# [x] Date ticks don't go on top if wg and len(st) == 1
# TODO I don't like the way NSLC labels are added; hard to read if too many traces or small figure

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime

from vdapseisutils.core.datasource.nslcutils import getNSLCstr


def __define_t_ticks__(t1, t2, ax, tick_type="datetime"):
    """
    Returns appropriate locations (in original axis coordinates) and labels for an x axis that is meant to span t1 to t2

    :param t1:
    :param t2:
    :param ax:
    :param tick_type:
    :return:
    """

    # input
    t1 = UTCDateTime(t1)  # convert t1 to UTCDateTime
    t2 = UTCDateTime(t2)  # convet t2 to UTCDateTime
    t0 = t1  # preserve the original t1

    if tick_type == "datetime":

        xlim = ax.get_xlim()  # original extent of x axis
        axis_length = xlim[1]-xlim[0]  # length of original axis
        plot_duration_min = (t2-t1)/60  # duration in minutes that axis is meant to represent
        axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes
        # plot_duration_samp = 2400  # axis_length --> ax2.get_xlim()

        # determine axis format based on length of time represented
        # there's gotta be a better way to do this
        tick_format = "%H:%M"
        tick_format_0 = "%Y/%m/%d"
        if plot_duration_min <= 2:  # 2 minutes or less
            nticks = 6
            tick_format = '%H:%M:%S'
            t1 = pd.Timestamp(t1.datetime).round("10s")
            t2 = pd.Timestamp(t2.datetime).round("10s")
        elif plot_duration_min <= 60:  # 2 minutes to 1 hour
            nticks = 6
            tick_format = "%H:%M"
            t1 = pd.Timestamp(t1.datetime).round("10T")
            t2 = pd.Timestamp(t2.datetime).round("10T")
        elif plot_duration_min <= 60*24:  # 1 hour to 1 day
            nticks = 7
            tick_format = "%H"
            t1 = pd.Timestamp(t1.datetime).round("1H")
            t2 = pd.Timestamp(t2.datetime).round("1H")
        elif plot_duration_min <= 60*24*7:  # 1 day to 1 week
            tick_format_0 = "%Y/%m/%d"
            tick_format = "%m/%d"
            t1 = pd.Timestamp(t1.datetime).round("1D")
            t2 = pd.Timestamp(t2.datetime).round("1D")

        dr = pd.date_range(t1, t2, periods=nticks)

        tick_labels = [d.strftime(tick_format) for d in dr]
        tick_labels[0] = dr[0].strftime(tick_format_0)

        # ticks need to be determined based on difference of t0.stats.starttime -
        tick0 = (UTCDateTime(dr[0])-t0) * axis_samp_rate
        ticks = np.linspace(0+tick0, axis_length+tick0, nticks)

    elif tick_type == "relative":
        raise Exception("Relative t tick labels not yet supported :-(")

    return ticks, tick_labels


class Clipboard(plt.Figure):

    def __init__(self, mode="wg",
                 figsize=None,
                 spectrogram={"log": False, "samp_rate": 25, "dbscale": True, "per_lap": 0.5, "mult": 25.0, "wlen": 6,
                              "cmap": "inferno"},
                 w_ax={"color": "k"},
                 g_ax={"ylim": [0.1, 10.0]},
                 s_ax={"color": "k", "mode": "loglog", "ylim": [1.0, 10.0]},
                 **kwargs,
                 ):


        # self.st = Stream(st.copy())  # Ensure object is Stream (converts Trace)
        self.st = Stream()
        self.ntr = 0
        self.n_subplots = 0
        self.subplots = []
        self.mode = mode

        # Figure object
        # self.figsize = (8.5, np.max([len(self.st)*2, 3])) or figsize
        self.figsize = (8.5, 10) or figsize

        # Other settings
        self.wg_ratio = [1, 3]  # height ratio of waveform axis v specgram axis

        # Default axis mode settings
        self.g_ax = g_ax
        self.w_ax = w_ax
        self.s_ax = s_ax

        super().__init__(figsize=self.figsize, **kwargs)


    def add_streams(self, st, mode="wg", **kwargs):

        self.st = Stream(st)
        self.ntr += len(self.st)
        self.nax = 2 if self.mode == "wg" else 1
        self.n_subplots += self.nax * self.ntr  # Total number of axes for given streams
        self.mode = mode
        gs = self.add_gridspec(nrows=self.n_subplots, ncols=1, height_ratios=self.wg_ratio * self.ntr)

        for i in range(self.ntr):
            axn = i*self.nax  # axis number
            tr = self.st[i]

            # Plot the top/secondary axis for waveforms if "wg"
            if self.mode == "wg":
                 self.subplots.append(self.add_subplot(gs[axn]))  # Create subplot for waveform
                 self.subplots[axn].plot(tr.data, **self.w_ax)  # Just plot points, not time
                 self.subplots[axn].set_xlim([0, len(tr.data)])
                 self.subplots[axn].yaxis.set_ticks_position('right')
                 ticks, tick_labels = __define_t_ticks__(tr.stats.starttime, tr.stats.endtime, self.subplots[axn])
                 self.subplots[axn].set_xticks(ticks, tick_labels, fontsize=12)

            # Plot the Primary axis
            if self.mode == "wg" or self.mode == "g":
                axn = axn+1 if self.mode == "wg" else axn  # increment axn by 1 if this is the second plot per trace
                self.subplots.append(self.add_subplot(gs[axn]))  # Create subplot for spectrogram
                self.subplots[axn].yaxis.set_ticks_position('right')
                self.subplots[axn].text(
                    -0.01, 0.67, getNSLCstr(tr), transform=self.subplots[axn].transAxes, rotation='vertical',
                    horizontalalignment='right', verticalalignment='center', fontsize=12,
                )
                tr.spectrogram(axes=self.subplots[axn], cmap="plasma")  # No spectrogram settings for now
                self.subplots[axn].set_ylim(self.g_ax["ylim"])
                ticks, tick_labels = __define_t_ticks__(tr.stats.starttime, tr.stats.endtime, self.subplots[axn])
                self.subplots[axn].set_xticks(ticks, tick_labels, fontsize=12)
            elif self.mode == "w":
                self.subplots.append(self.add_subplot(gs[axn]))  # Create subplot for waveform
                self.subplots(tr.data, **self.w_ax)  # Just plot points, not time
                self.subplots[axn].set_xlim([0, len(tr.data)])
                self.subplots[axn].yaxis.set_ticks_position('right')
                ticks, tick_labels = __define_t_ticks__(tr.stats.starttime, tr.stats.endtime, self.subplots[axn])
                self.subplots[axn].set_xticks(ticks, tick_labels, fontsize=12)
            self.subplots[axn].text(
                -0.01, 0.5, getNSLCstr(tr), transform=self.subplots[axn].transAxes, rotation='vertical',
                horizontalalignment='right', verticalalignment='center', fontsize=12,
            )

            # Put axes on top of top plot
            # Remove middle axes
            if len(self.figure.get_axes()) >= 2:  # if len>2 bc 1 st will produce two axes
                if len(self.figure.get_axes()) == 2:
                    self.figure.get_axes()[0] = self.figure.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
                for i in range(1, len(self.figure.get_axes())-1):  # Remove middle axes
                    self.figure.get_axes()[i] = self.figure.get_axes()[i].set_xticks([])

        print("Making axes...")

    def __plot_streams(self):
        # for i in range(len(self.subplots)):
        for i, subplot in enumerate(self.subplots):
            print(i)
            print(self.subplots[i])
            print(self.st[i])
            print()
            subplot.plot(self.st[i].data)

    def plot_point(self, data_point, *args, **kwargs):
        for i, subplot in enumerate(self.subplots):
            subplot.plot(data_point, 0, *args, **kwargs)

