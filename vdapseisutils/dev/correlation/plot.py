import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from obspy.imaging.util import _set_xaxis_obspy_dates

# [x] Add ax=None args to plot routines
# [x] Remove filename args (keep these methods general, to the core point)
# [x] Always return ax object
# TODO plot_fi(datetimes, fis, ...)  # make a scatter plot w custom colorbar, etc
"""
How to limit the range of a colormap:
    cmap = cm.get_cmap('inferno')
    new_cmap = cmap(np.linspace(0.15, 0.85, 256))
    new_cmap = cm.colors.ListedColormap(new_cmap)
"""

def cmapmpl(ccthresh):
    """Creates a colormap with inferno_r where values below the cc threshold have an alpha value of 0.25?"""

    n1 = int(np.ceil(256*(ccthresh)))  # Number of colors to represent below threshold
    colors = plt.cm.inferno_r(np.linspace(0.0, 1.0, 256))  # take n2 samples from ccthresh to 1
    colors[0:n1, 3] = np.zeros((1, n1))+0.25
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return cmap

# TODO Integers as xticklabels and yticklabels
# TODO Duplicate shifmatrixmpl with different defaults
def ccmatrixmpl(ccm, vmin=0.0, vmax=1.0, cmap='inferno', title="Cross Correlation Matrix", ax=None):
    """Plots a cross correlation matrix"""

    # Initialize fig and ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))  # Create a new figure and axes if ax is not provided
    else:
        fig = ax.figure  # Use the figure associated with the provided axes

    # Set vmin and vmax if they are None
    if vmin is None:
        vmin = np.nanmin(ccm)  # Use nanmin to ignore NaNs
    if vmax is None:
        vmax = np.nanmax(ccm)  # Use nanmax to ignore NaNs

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = ax.imshow(ccm, norm=norm, cmap=cmap)

    # Adjust the colorbar to match the height of the axes
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)  # Adjust fraction and pad as needed

    ax.set_title(title)  # Use ax to set title
    ax.set_ylabel("Event #")
    ax.set_xlabel("Event #")

    return ax


def cchistmpl(ccm, nbins=20, facecolor="Purple", alpha=0.5, ax=None):
    """Plots a cross correlation histogram"""

    # Plot cross-correlation histogram
    if not ax:
        fig, ax = plt.subplots()
    A = ccm[np.tril_indices(ccm.shape[0], k=-1)]  # returns just the lower part of the matrix
    n, bins, patches = plt.hist(A, nbins, facecolor=facecolor, alpha=alpha)
    plt.xlim(0.0, 1.0)

    return ax


# [x] Add line_kwargs and box_kwargs
# [x] Add cmap="inferno" arg
# TODO Should BoundaryNorm be hard-coded, or not?
# [x] Use time values as x
# [x] min_members=2
# [x] Some controlling of yticklabels (incase families are provided "out of order")
#       Or maybe it is easier to have sort="chron" as an option here
# [x] Add filter and sort methods to this method? Removes need for one-off variables elsewhere
# TODO bin_size is not working correctly
def plot_timeline(data, bin_size="1D", boxes=True, cmap="inferno", nl = 100,
                          family_id=None, sort=None, min_members=None,
                          line_kwargs={'color': 'black', 'linewidth': 1},
                          box_kwargs={'edgecolor': 'black'},
                          ax=None):
    import pandas as pd
    import numpy as np
    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.dates import date2num

    # Convert bin_size to float (fraction of a day)
    bin_size_float = width = pd.to_timedelta(bin_size).total_seconds() / (86400.)

    # Initialize family id, if not provided
    family_id = np.arange(len(data)) if family_id is None else np.array(
        family_id)  # auto-set family idx if not provided

    from vdapseisutils.dev.correlation.families import filter_and_sort
    if min_members:
        data, family_id = filter_and_sort(data, min_members=min_members, family_id=family_id)
    if sort:
        data, family_id = filter_and_sort(data, order=sort, family_id=family_id)

    # Create yticks and yticklabels (everything is already sorted and filtered)
    yticks = np.arange(len(data))
    yticklabels = [f'Family {i: 3d}' for i in family_id]  # e.g., 'Family   1'

    # Determine the earliest and latest dates
    all_dates = [date for sublist in data for date in sublist]
    earliest_date = min(all_dates)
    latest_date = max(all_dates)

    # Calculate the rounded start date
    rounded_start_date = earliest_date.date()
    rounded_latest_date = latest_date.date()
    # If bin_size is less than 1 day, make sure the daterange has a different start/stop
    # This handles cases where only 1 day of data is supplied
    if rounded_latest_date == rounded_start_date and pd.to_timedelta(bin_size) < pd.to_timedelta("1D"):
        rounded_latest_date += dt.timedelta(days=1)

    # Create a date range from the rounded start date to the latest date
    date_range = pd.date_range(start=rounded_start_date, end=rounded_latest_date, freq=bin_size)

    # Initialize a dictionary to hold overall binned counts
    overall_binned_counts = {date: 0 for date in date_range}

    # PRE-COMPUTE ALL BINNING AND COUNTS
    # Store binned data for each family
    all_binned_data = []
    max_count = 0  # Track the maximum count across all bins

    # Process each list of datetimes
    for datetimes in data:
        # Convert to pandas Series for easier manipulation
        series = pd.Series(datetimes)

        # Count occurrences for each date in the date range
        binned = series.dt.floor(bin_size).value_counts().reindex(date_range, fill_value=0)

        # Keep track of the maximum count for colormap normalization
        if binned.max() > max_count:
            max_count = binned.max()

        # Store binned data for later use
        all_binned_data.append(binned)

        # Update overall binned counts
        for date, count in binned.items():
            overall_binned_counts[date] += count

    # Create a discrete colormap based on the actual maximum count
    num_levels = ((max_count + (nl-1)) // nl) * nl  # round up to nearest 100 (w/out importing math.ceil())
    dcmap = plt.get_cmap(cmap, num_levels)  # Use actual max count for discrete colors
    boundaries = np.arange(1, num_levels + 2)  # Add 1 more for proper binning
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=num_levels)

    # Create a figure and axis
    if not ax:
        fig, ax = plt.subplots(figsize=(10, len(data)))

    # NOW DO THE PLOTTING
    for y, binned in enumerate(all_binned_data):
        print()

        # Draw a line from the first to the last occurrence
        if not binned.empty:
            first_idx = np.argmax(binned.values != 0)  # first ocurrence within family
            last_idx = len(binned.values) - 1 - np.argmax(binned.values[::-1] != 0)  # last occurrence within family
            first = binned.index[first_idx]
            last = binned.index[last_idx]

            # Plot line for extent
            # Extend plot -0.5:+0.5 mdates so that line is centered on the xtick labels
            ax.plot([date2num(first) - width / 2, date2num(last) + width / 2], [y, y], zorder=-1,
                    marker="|", **line_kwargs)  # color='black', linewidth=1,

            # Plot rectangles for each bin
            if boxes:
                for date, count in binned.items():
                    if count > 0:  # Only plot if count is greater than 0
                        color = dcmap(norm(count))
                        # Plot rectange for date. Subtract 0.5 mdates so that box is centered on xtick label
                        ax.add_patch(plt.Rectangle((date2num(date) - width / 2, y - 0.4), width, 0.8,
                                                   facecolor=color, **box_kwargs))

                # Annotate the total count at the end of the line
                total_count = binned.sum()
                x = date2num(last)
                xtext = date2num(last) + width * 1.1  # Add 1.1 bin_size in matplotlib mdates units
                ax.annotate(int(total_count), xy=(x, y), xytext=(xtext, y),
                            fontsize=10, verticalalignment='center')

    # Set the y-ticks
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Set x-ticks to be the overall binned dates
    ax.set_xticks([date2num(d) for d in date_range])
    from obspy.imaging.util import \
        _set_xaxis_obspy_dates  # get the xticklabels in the correct time object, and this will work nicely
    _set_xaxis_obspy_dates(ax)

    # Add a colorbar below the axes
    if boxes:
        sm = plt.cm.ScalarMappable(cmap=dcmap, norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, aspect=50)
        cbar.set_label('Count of Occurrences')
        plt.subplots_adjust(top=0.85, bottom=0.15)  # Adjust layout to make room for the colorbar

    return ax

# [x] Make it so that input arguments are coerced to a list-like of datetime objects
# ? Use num2date for matplotlib?
# [x] Switch to Total Seismicity and Repeaters
# TODO Add units="counts" | "percentage"
# TODO ? Add plot_type="bar" | "step" | "plot", etc. & plot_kwargs=None
def plot_rate(repeaters, events=None, bin_size="1D", ax=None):
    """
    Plot event rates for repeaters and optionally orphans.

    Parameters:
    -----------
    repeaters : list
        List of datetime.datetime objects representing repeater events
    events : list, optional
        List of datetime.datetime objects representing all events
    bin_size : str, default="1D"
        Bin size for aggregating events (e.g., "1D" for daily, "1H" for hourly)
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes will be created

    Returns:
    --------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import datetime as dt

    # Convert bin_size to float (fraction of a day)
    bin_size_float = width = pd.to_timedelta(bin_size).total_seconds() / (86400.)

    # Ensure repeaters is a flat list of datetime objects
    if not repeaters:
        repeaters = []
    elif not isinstance(repeaters, list):
        try:
            repeaters = list(repeaters)
        except:
            raise TypeError("repeaters must be convertible to a list")

    # Handle optional orphans parameter
    if events is None:
        events = []
    else:
        # Ensure orphans is a flat list of datetime objects
        if not isinstance(events, list):
            try:
                events = list(events)
            except:
                raise TypeError("orphans must be convertible to a list")

    # Validate that all items are datetime objects
    for i, event in enumerate(repeaters):
        if not isinstance(event, dt.datetime):
            raise TypeError(f"repeaters[{i}] is not a datetime.datetime object")

    for i, event in enumerate(events):
        if not isinstance(event, dt.datetime):
            raise TypeError(f"orphans[{i}] is not a datetime.datetime object")

    # Handle empty lists case
    if not repeaters and not events:
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_ylabel("Event Counts")
        ax.text(0.5, 0.5, "No events to display", ha='center', transform=ax.transAxes)
        return ax

    # Determine the earliest and latest dates
    if repeaters and events:
        earliest_date = min(min(repeaters), min(events))
        latest_date = max(max(repeaters), max(events))
    elif repeaters:
        earliest_date = min(repeaters)
        latest_date = max(repeaters)
    else:  # only orphans exist
        earliest_date = min(events)
        latest_date = max(events)

    # Calculate the rounded start date
    rounded_start_date = earliest_date.date()
    rounded_latest_date = latest_date.date()

    # If bin_size is less than 1 day, make sure the daterange has a different start/stop
    # This handles cases where only 1 day of data is supplied
    if rounded_latest_date == rounded_start_date and pd.to_timedelta(bin_size) < pd.to_timedelta("1D"):
        rounded_latest_date += dt.timedelta(days=1)

    # Create a date range from the rounded start date to the latest date
    date_range = pd.date_range(start=rounded_start_date, end=rounded_latest_date, freq=bin_size)

    # Create a figure and axis
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Convert to pandas Series
    series_r = pd.Series(repeaters)

    # Count occurrences for each date in the date range
    binned_r = series_r.dt.floor(bin_size).value_counts().reindex(date_range, fill_value=0)

    if events:
        series_o = pd.Series(events)
        binned_o = series_o.dt.floor(bin_size).value_counts().reindex(date_range, fill_value=0)
        # Plot total (repeaters + orphans)
        ax.bar(binned_o.index, binned_o.values, width=width, facecolor='black', edgecolor="black", alpha=0.85,
               label="Total")
    # else:
    #     binned_o = pd.Series([], dtype='datetime64[ns]').value_counts().reindex(date_range, fill_value=0)

    # Plot repeaters
    ax.bar(binned_r.index, binned_r.values, width=width, facecolor='red', edgecolor="black", alpha=0.85, label="Repeaters")

    ax.legend(loc='upper left')
    ax.set_ylabel("Event Counts")

    # Call external function if available, otherwise set basic date formatting
    try:
        _set_xaxis_obspy_dates(ax)
    except NameError:
        ax.figure.autofmt_xdate()

    return ax

# TODO Switch to Total Seismicity and Repeaters
# TODO Add units="counts" | "percentage"
# TODO ? Add plot_type="bar" | "step" | "plot", etc. & plot_kwargs=None
def plot_rate_v001(repeaters, orphans, bin_size="1D", ax=None):
    """Repeaeters and orphans should be lists of datetime.datetime objects"""

    import matplotlib.pyplot as plt
    import pandas as pd

    # Determine the earliest and latest dates
    earliest_date = min(min(repeaters), min(orphans))
    latest_date = max(max(repeaters), max(orphans))

    # Calculate the rounded start date
    rounded_start_date = earliest_date.date()
    rounded_latest_date = latest_date.date()

    # Create a date range from the rounded start date to the latest date
    date_range = pd.date_range(start=rounded_start_date, end=rounded_latest_date, freq=bin_size)

    # Create a figure and axis
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 4))

    series_r = pd.Series(repeaters)
    series_o = pd.Series(orphans)

    # Count occurrences for each date in the date range
    binned_r = series_r.dt.floor(bin_size).value_counts().reindex(date_range, fill_value=0)
    binned_o = series_o.dt.floor(bin_size).value_counts().reindex(date_range, fill_value=0)

    # ax.stairs(binned_r.values[1:], binned_r.index, color='red', linewidth=1, zorder=-1)
    # # ax.stairs(binned_o.values[1:], binned_o.index, color='black', linewidth=1, zorder=-1)
    # ax.stairs(binned_o.values[1:] + binned_r.values[1:], binned_o.index, color='green', linewidth=1, zorder=-1)

    # ax.step(binned_r.index, binned_r.values, where="post", color='red', linewidth=1, zorder=-1)
    # ax.step(binned_r.index, binned_r.values + binned_o.values, where="post", color='black', linewidth=2, zorder=-1)
    # ax.step(binned_r.index, binned_r.values, where="post", color='red', linewidth=1, zorder=-1)

    # ax.step(binned_r.index, binned_r.values, where="post", color='red', linewidth=1, zorder=-1)
    ax.bar(binned_r.index, binned_r.values + binned_o.values, facecolor='black', edgecolor="black", alpha=0.85,
           label="Total")
    ax.bar(binned_r.index, binned_r.values, facecolor='red', edgecolor="black", alpha=0.85, label="Repeaters")
    ax.legend(loc='upper left')
    ax.set_ylabel("Event Counts")
    _set_xaxis_obspy_dates(ax)

    return ax

def plot_fi(datetimes, fis, ax=None, **kwargs):
    print("Not yet implemented.")


def plot_familywaves(streams, main=None, type=None, order="descending", spacing=0.75, max_events=25, cmap="RdYlBu", ax=None):
    """Plots all waveforms from a family on a single axes. Uses imshow for more than N events

    streams: list of ObsPy Streams
    main: Stream object
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(8,3))

    if order.lower() == "descending":
        streams.reverse()
    elif order.lower() == "ascending":
        pass
    else:
        print("Order must be 'descending' or 'ascending'. Doing nothing.")

    # Put traces into NumPy array
    data = []
    for st in streams:
        data.append(st[0].data / np.max(np.abs(st[0].data)))
    max_length = max(len(arr) for arr in data)
    trace_data = np.array(
        [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in data])

    # Create array for y values
    y = np.arange(trace_data.shape[0])  # This will create an array [0, 1, ..., n] for n rows
    y_values = y[:, np.newaxis]  # Reshape y to be a column vector

    # Plot
    if (len(streams) <= max_events) or (type == "plot"):
        for i in range(trace_data.shape[0]):
            ax.plot(range(trace_data.shape[1]), trace_data[i]+i*spacing, "rebeccapurple", lw=0.5)
            if main:
                ymain = y_values[0]
                maindata = main[0].data / np.max(np.abs(main[0].data))
                ax.plot(range(maindata.shape[0]), maindata + ymain, "deeppink", lw=2.5, zorder=-10)
            ylim = [-1, trace_data.shape[0]*spacing]
    elif (len(streams) > max_events) or (type == "imshow"):
        ax.imshow(trace_data, cmap=cmap, aspect="auto", interpolation="none")
        ylim = [-0.4, trace_data.shape[0]-0.4]

    # Configure axes
    ax.set_ylim(ylim)  # trace_data.shape[0] is # of rows
    ax.set_xlim([0, trace_data.shape[1]])
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
