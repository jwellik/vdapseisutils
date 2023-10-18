"""
Swarm like plots for matplotlib

Author: Jay Wellik
Last updated: 2022 June 30
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from vdapseisutils.core.datasource.nslcutils import getNSLCstr


def __define_t_ticks__(t1, t2, ax, tick_type="datetime"):

    # input
    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2)
    t0 = t1

    if tick_type == "datetime":

        xlim = ax.get_xlim()
        axis_length = xlim[1]-xlim[0]
        plot_duration_min = (t2-t1)/60
        axis_samp_rate = axis_length / (plot_duration_min * 60)
        # plot_duration_samp = 2400  # axis_length --> ax2.get_xlim()

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

    def __init__(self, st, mode="wg", taxis="datetime",
                 spectrogram={"log":False, "samp_rate":25, "dbscale":True, "per_lap":0.5, "mult":25.0, "wlen":6, "cmap":"plasma"},
                 wax={"color":"k"},
                 gax={"ylim":[0.1, 10]},
                 sax={"mode":"loglog", "ylim":[1.0, 10.0]}):

        self.st = st.copy()
        self.ntr = len(st)
        self.mode = mode
        self.set_spectrogram_settings(**spectrogram)  # Sets self.spectrogram_settings
        self.set_wax(**wax)  # Sets self.wax
        self.set_gax(**gax)  # Sets self.gax
        self.set_sax(**sax)  # Sets self.sax
        # self.__plot_data__(st, ax = self.ax)

    def set_spectrogram_settings(self, **kwargs):
        self.spectrogram_settings = dict(**kwargs)

    def set_wax(self, **kwargs):
        self.wax = dict(**kwargs)

    def set_gax(self, **kwargs):
        self.gax = dict(**kwargs)

    def set_sax(self, **kwargs):
        self.sax = dict(**kwargs)

    def plot(self):

        fig = plt.figure(figsize=(8,10.5), constrained_layout=False)

        self.nax = 2 if self.mode=="wg" else 1  # number of axes per trace
        num_rows = self.ntr*self.nax
        height_ratio = [1, 3] if self.mode=="wg" else [1]  # w:g ratio is 1:3 if necessary
        gs = fig.add_gridspec(nrows=num_rows, ncols=1, height_ratios=height_ratio * self.ntr)
        print(gs, self.ntr)

        for i in range(self.ntr):
            tr = self.st[i]
            print(i, tr)

            # ax = fig.add_subplot(gs[0])  # Create subplot for Waveform/Spectrogram/Spectra
            # Plot the top/secondary axis if "wg"
            if self.mode == "wg":
                # print(i)
                axT = fig.add_subplot(gs[i*self.nax])  # Create subplot for waveform
                axT.plot(tr.data, **self.wax)  # Just plot points, not time
                axT.set_xlim([0, len(tr.data)])
                axT.yaxis.set_ticks_position('right')
                ticks, tick_labels = __define_t_ticks__(tr.stats.starttime, tr.stats.endtime, axT)
                axT.set_xticks(ticks, tick_labels, fontsize=12)

            # Plot the Primary axis
            if self.mode == "wg" or self.mode == "g":
                # print(i)
                a = 1 if self.mode == "wg" else 0
                axP = fig.add_subplot(gs[i*self.nax+a])  # Create subplot for spectrogram
                axP.yaxis.set_ticks_position('right')
                # axP.text(
                #     -0.01, 0.67, getNSLCstr(tr), transform=axP.transAxes, rotation='vertical',
                #     horizontalalignment='right', verticalalignment='center', fontsize=12,
                # )
                axP = tr.spectrogram(axes=axP, **self.spectrogram_settings)
                axP.set_ylim(self.gax["ylim"])
                ticks, tick_labels = __define_t_ticks__(tr.stats.starttime, tr.stats.endtime, axP)
                axP.set_xticks(ticks, tick_labels, fontsize=12)
            elif self.mode == "w":
                # print(i)
                axP = fig.add_subplot(gs[i*self.nax])  # Create subplot for waveform
                axP.plot(tr.data, **self.wax)  # Just plot points, not time
                axP.set_xlim([0, len(tr.data)])
                axP.yaxis.set_ticks_position('right')
                ticks, tick_labels = __define_t_ticks__(tr.stats.starttime, tr.stats.endtime, axP)
                axP.set_xticks(ticks, tick_labels, fontsize=12)
            axP.text(
                -0.01, 0.5, getNSLCstr(tr), transform=axP.transAxes, rotation='vertical',
                horizontalalignment='right', verticalalignment='center', fontsize=12,
            )

            # Put axes on top of top plot
            # Remove middle axes
            if len(fig.get_axes()) > 2:  # if len>2 bc 1 stream will produce two axes
                fig.get_axes()[0] = fig.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
                for i in range(1, len(fig.get_axes())-1):  # Remove middle axes
                    fig.get_axes()[i] = fig.get_axes()[i].set_xticks([])

        return fig

    def axvline(self, t):
        pass

    def show(self):
        plt.show()


def plot_triggers(triggers, cft, st, cftlim=None,
                  trigonoff=(0.6, 1.5), nstatrig=None,
                  stalta=None,
                  wylim=None, wave_color='k'):
    """SWARMWG Plots waveform/spectrogram pairs for Stream objects with multiple Traces

    ARGS
    st          : Stream : Stream object of seismic waveform
    cft         : Stream : Stream object of cft function

    KWARGS
    cftlim      : list-like : 2 element list of Y-axis limits for spectrogram (Hz)
                  Default:[1,10]
    wylim       : list-like : 2 element list of Y-axis limits for waveform (data units)
                  Default: None (auto-scale)
    wave_color  : Color of the waveform. Anything understood by Matplotlib as a color
    """

    import numpy as np

    #     st = replaceGapValue(st, gap_value=np.nan, fill_value=0) # Not sure this code is working

    nstreams = len(st)
    #    plot_duration = st[0].stats.endtime - st[0].stats.starttime
    figheight = 0.5 + 2.5 * nstreams
    left = 0.05
    right = 0.95
    top = 0.10
    bottom = 0.05
    space = 0.03
    wgratio = [1, 3]
    axheight = (1.0 - top - bottom - space * (nstreams - 1)) / nstreams
    suptitley = 0.975
    suptitlefs = 12

    fig = plt.figure(figsize=(12, figheight), constrained_layout=False)

    for n in range(len(st)):
        tr = st[n]
        tr2 = cft[n]

        # bottom to top
        # axbottom = bottom+axheight*n+space*(n)
        # axtop    = axbottom+axheight
        # top to bottom
        axtop = 1 - top - axheight * n - space * n
        axbottom = axtop - axheight
        # print(n, axbottom, axtop)

        wg = fig.add_gridspec(
            nrows=wgratio[1] + 1, ncols=1, left=left, right=right,
            # nrows=2, ncols=1, left=left, right=right,
            bottom=axbottom, top=axtop,
            wspace=0.00, hspace=0.00
        )

        # Create subplots for two axes per station
        w = fig.add_subplot(wg[0, :])  # Create subplot for Waveform
        g = fig.add_subplot(wg[1:, :])  # Create subplot for Spectrogram

        # First, plot coincidence_triggers
        # {'time': 2004 - 09 - 28T00: 00:11.360200
        # Z,
        # 'stations': ['SEP', 'HSR'],
        # 'trace_ids': ['UW.SEP..EHZ', 'UW.HSR..EHZ'],
        # 'coincidence_sum': 2.0,
        # 'similarity': {},
        # 'duration': 13.940000057220459},
        if nstatrig is None: nstatrig = len(st)
        for trig in triggers:
            if trig["coincidence_sum"] >= nstatrig:
                t1 = UTCDateTime(trig["time"]).matplotlib_date
                t2 = UTCDateTime(trig["time"]+trig["duration"]).matplotlib_date
                w.axvspan(t1, t2, facecolor='y', alpha=0.5)
                g.axvspan(t1, t2, facecolor='y', alpha=0.5)

        w.plot(tr.times("matplotlib"), tr.data, color=wave_color)
        w.set_xlim([tr.times("matplotlib")[0], tr.times("matplotlib")[-1]])
        # if wylim is not None: w.set_ylim(wylim)
        w.yaxis.set_ticks_position('right')

        g.yaxis.set_ticks_position('right')
        g.text(
            -0.01, 0.67, getNSLCstr(tr), transform=g.transAxes, rotation='vertical',
            horizontalalignment='right', verticalalignment='center', fontsize=12,
        )
        g.plot(tr2.times("matplotlib"), tr2.data, color=wave_color)
        g.set_xlim([tr.times("matplotlib")[0], tr.times("matplotlib")[-1]])  # Use xlim of waveform just to be sure
        g.axhline(y=trigonoff[0], color="b", linestyle=":")
        g.axhline(y=trigonoff[1], color="r", linestyle=":")
        if cftlim is not None: g.set_ylim(cftlim)

    # Set xaxis ticks and labels
    # ??? settaxis( fig, n=2, minticks=3, maxticks=7 ) # n is how many plots per stream
    """
    The swarmwg() function makes two axes for each waveform - a waveform axis (w)
    and a spectrogram axis (g).
    First, 'locator' automatically decides how many xticks to create. Odd numbers ensure
    that no ticks are at the end of the graphs.
    Second, 'formatter' automatically decides the datetime format, based on how
    'zoomed in' the plot is.
    At the end, the last axis always gets x ticks.
    Before that - in the if/for loop - if there is more than 1 waveform (2 axes), xticks
    are added to the top axis, and all other axes are removed.
    This allows the plots to sit right on top of each other with xticks on the bottom
    (and top if there are multiple waveforms).
    """
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)  # Dynamic choice of xticks
    formatter = mdates.ConciseDateFormatter(locator)  # Automatic date formatter
    if len(fig.get_axes()) > 2:  # if len>2 bc 1 stream will produce two axes
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_locator(locator)  # Format top axis
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_formatter(formatter)
        for n in range(1, len(fig.get_axes())):  # Remove middle axes
            fig.get_axes()[n] = fig.get_axes()[n].set_xticks([])
    fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_locator(
        locator)  # Format bottom axis (bottom axis is always [-1])
    fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_formatter(formatter)

    # trigger header
    nstatrig = 0 if not nstatrig else nstatrig
    stalta = ["NA", "NA"] if not stalta else stalta
    title_txt_1 = "# Stations : {}/{} ({} triggers)".format(nstatrig, len(st), len(triggers))
    title_txt_2 = "STA/LTA : {}/{}     ON/OFF : {}/{}".format(stalta[0], stalta[1], trigonoff[0], trigonoff[1])
    plt.suptitle("{:^100}\n{:^100}".format(title_txt_1, title_txt_2), y=suptitley, fontsize=suptitlefs)

    return fig

# RESOURCES:
# https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
# "offset_formats" are the big dates on the bottom right of the axis
