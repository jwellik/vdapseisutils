def plot_clipboard(st, tick_type="time", figsize=(10, 3), height_ratios=[0.2, 0.8],
                   alim=None,
                   flim=None, overlap=0.86, wlen=2.0, dbscale=True, log_power=False, cmap="viridis"):

    from vdapseisutils.core.swarmmpl.clipboard import plot_wave, plot_spectrogram
    from obspy.imaging.util import _set_xaxis_obspy_dates
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import datetime

    # Create a figure with a specific size
    fig = plt.figure(figsize=figsize)

    # Create a gridspec with height ratios
    gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)  # Top: 20%, Bottom: 80%

    # Create the first subplot (top)
    ax0 = fig.add_subplot(gs[0])
    tr = st[0].copy()
    ax0 = plot_wave(tr, ax=ax0)  # label="AV.GAEA.--.BHZ"
    ax0.set_xlim(tr.times("utcdatetime")[0].datetime, tr.times("utcdatetime")[-1].datetime)
    if alim is not None:
        ax0.set_ylim(alim)
    # ax0.set_ylabel("Counts")
    ax0.set_xticks([])

    # Create the second subplot (bottom)
    ax1 = fig.add_subplot(gs[1])
    ax1, data = plot_spectrogram(st[0], overlap=overlap, wlen=wlen, dbscale=dbscale, log_power=log_power, cmap=cmap, ax=ax1)
    ax1.set_xlim(st[0].times("utcdatetime")[0].datetime, st[0].times("utcdatetime")[-1].datetime)
    ax1.set_ylabel(st[0].id)
    if flim is not None:
        ax1.set_ylim(flim)

    # Prune yticklabels
    from matplotlib.ticker import MaxNLocator
    # ax0.yaxis.set_major_locator(MaxNLocator(nbins=3))  # "upper" "lower" "both"
    # ax1.yaxis.set_major_locator(MaxNLocator(prune='upper'))  # "upper" "lower" "both"
    ax0.set_yticks([])
    ax1.set_yticks(ax1.get_yticks())
    # ax1.set_ylabel("Frequency (Hz)")

    # Set xticks
    _set_xaxis_obspy_dates(ax1)  # set xticks as datetimes (always)
    if tick_type == "relative":
        from matplotlib.dates import num2date, date2num
        _set_xaxis_obspy_dates(ax1)  # set xticks as datetimes (always)
        xticks = ax1.get_xticks()  # matplotlib datenums
        nticks = len(xticks)  # original number of xticks
        xtick_dt = [num2date(x) for x in xticks]  # convert matplotlib datenum to datetime.datetime
        gaps = [xtick_dt[i+1] - xtick_dt[i] for i in range(len(xtick_dt) - 1)]  # xtick spacing (datetime.timedelta)
        tick_delta = sum(gaps, datetime.timedelta(0)) / len(gaps)  # not sure why sum includes timedelta(0)
        xticks_new_dt = [xtick_dt[0] + tick_delta * n for n in range(1,nticks)]  # datetime.datetime; range(1,nticks) removes first tick (should be 0)
        xticklabels = [str((tick_delta * n).seconds / 60) for n in range(1,nticks)]  # / 60 --> minutes
        xticklabels[-1] = xticklabels[-1] + " min"
        xticks_new = [date2num(x) for x in xticks_new_dt]  # matplotlib datenum
        ax1.set_xticks(xticks_new)
        ax1.set_xticklabels(xticklabels)

    # Adjust the layout
    plt.subplots_adjust(hspace=0.0)

    return fig
