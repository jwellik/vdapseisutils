import numpy as np
import datetime as dt
from pytz import timezone
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from vdapseisutils.sandbox.swarmmpl import Helicorder, greyscale_hex
#from vdapseisutils import Helicorder

def format_datetime_yticklabels(datetime_list):
    """Formats the first datetime of each day as %Y/%m/%d and the rest as %H:%M"""

    formatted_datetimes = []
    previous_date = None

    for dt in datetime_list:
        # Check if the current datetime is the first of the day
        if dt.date() != previous_date:
            formatted_datetimes.append(dt.strftime('%Y/%m/%d'))  # Format as %y/%m/%d
        else:
            formatted_datetimes.append(dt.strftime('%H:%M'))  # Format as %H:%M

        previous_date = dt.date()

    return formatted_datetimes


def dev_xy_labels():

    start = UTCDateTime("2021/05/20")
    stop = UTCDateTime("2021/05/21")
    client = Client("IRIS")
    print("Getting waveforms...")
    st = client.get_waveforms("II", "MBAR", "00", "BHZ", start, stop)
    interval = 60
    timezone_left = "UTC"
    timezone_right = "Asia/Singapore"

    heli_starttime = min([trace.stats.starttime for trace in st]).datetime.astimezone(timezone("UTC"))
    heli_endtime = max([trace.stats.endtime for trace in st]).datetime.astimezone(timezone("UTC"))

    tick_format_first = "%Y/%m/%d"  # format for yticklabel at top
    tick_format_last = "UTC%z"  # format for yticklabel at bottom
    tick_format = "%H:%M"  # format for all other yticklabels
    datalength = float((heli_endtime - heli_starttime) / dt.timedelta(seconds=60))  # helicorder length, in minutes
    line_length = float(interval)  # length of one horizontal line, in minutes
    n_lines = int(np.ceil(datalength / line_length))  # number of lines on the helicorder
    ylabel_spacing = int(4)  # number of lines per ylabel (in other words, add a label every nth line)
    ytick_lines = list(range(n_lines, 0, -1 * ylabel_spacing))  # indices of the lines that should have yticklabels
    ytick_lines.append(0)  # Add a yticklabel to the bottom line
    yticklabel_dt_l = [heli_starttime + i * dt.timedelta(minutes=line_length) for i in ytick_lines][::-1]  # yticklabels in datetime format
    yticklabel_dt_l = [t.astimezone(timezone(timezone_left)) for t in yticklabel_dt_l]  # adjust datetimes to proper timezone
    yticklabels_l = format_datetime_yticklabels(yticklabel_dt_l)
    # yticklabels_l[0] = yticklabel_dt_l[0].strftime(tick_format_first)  # change format of top yticklabel
    yticklabels_l[-1] = yticklabel_dt_l[-1].strftime(tick_format_last)  # change format of bottom yticklabel
    yticklabel_dt_r = [t + dt.timedelta(minutes=line_length) for t in yticklabel_dt_l]
    yticklabel_dt_r = [t.astimezone(timezone(timezone_right)) for t in yticklabel_dt_r]  # adjust datetimes to proper timezone
    yticklabels_r = format_datetime_yticklabels(yticklabel_dt_r)
    yticklabels_r[-1] = yticklabel_dt_r[-1].strftime(tick_format_last)  # change format of bottom yticklabel
    ytick_offset = -0.5  # E.g., Line 1 goes from 0 to 1, so the center is at -0.5
    ytick_pos = np.array(ytick_lines) + ytick_offset  # get y-axes value for yticks
    print("Done.")


def main():

    start = UTCDateTime("2021/05/20")
    stop = UTCDateTime("2021/05/21")
    interval = 120
    latitude = -1.52
    longitude = 29.25
    maxradiuskm = 50  # km
    minmag = 4.6

    client = Client("IRIS")
    print("Getting waveforms...")
    st = client.get_waveforms("II", "MBAR", "00", "BHZ", start, stop)
    st.write("goma_waveform.mseed")
    print("Getting events...")
    cat = client.get_events(starttime=start, endtime=stop,
                            # latitude=latitude, longitude=longitude, maxradius=maxradiuskm/110,
                            minmagnitude=minmag
                            )

    # print("Creating plot... (60 minute interval)")
    # h = Helicorder(st, interval=60, color=greyscale_hex,  # define the helicorder specs
    #                show_y_UTC_label=True, right_vertical_labels=True,
    #                title=st[0].id + ": Earthquakes at Goma",
    #                )
    # h.plot()  # plot the st
    # h.plot_tags(UTCDateTime("2021/05/20 01:36:30"), color="yellow", markersize=8)  # plot a single time
    # h.plot_tags(UTCDateTime("2021/05/20 17:19:35"), marker="|", markeredgecolor="blue", markersize=15)  # plot a single time as a P arrival
    # h.plot_tags(UTCDateTime("2021/05/20 17:25:00"), marker="|", markeredgecolor="red", markersize=15)   # plot a single time as a S arrival
    # h.plot_tags(UTCDateTime("2021/05/20 17:44:45"), marker="|", markeredgecolor="black", markersize=15)  # plot a single time as a coda end
    # h.plot_tags([UTCDateTime("2021/05/20 08:31:00"), "2021/05/20 02:08:00"], color="yellow")  # plot a list of times given in any format
    #
    # plt.show()  # show the helicorder

    # print("Creating plot... (30 minute interval)")
    # h = Helicorder(st, interval=30, color=greyscale_hex,  # define the helicorder specs
    #                show_y_UTC_label=True, right_vertical_labels=True,
    #                title=st[0].id + ": Earthquakes at Goma",
    #                )
    # h.plot()  # plot the st
    # h.plot_tags(UTCDateTime("2021/05/20 01:36:30"), color="yellow", markersize=8)  # plot a single time
    # h.plot_tags(UTCDateTime("2021/05/20 17:19:35"), marker="|", markeredgecolor="blue", markersize=15)  # plot a single time as a P arrival
    # h.plot_tags(UTCDateTime("2021/05/20 17:25:00"), marker="|", markeredgecolor="red", markersize=15)   # plot a single time as a S arrival
    # h.plot_tags(UTCDateTime("2021/05/20 17:44:45"), marker="|", markeredgecolor="black", markersize=15)  # plot a single time as a coda end
    # h.plot_tags([UTCDateTime("2021/05/20 08:31:00"), "2021/05/20 02:08:00"], color="yellow")  # plot a list of times given in any format
    #
    # plt.show()  # show the helicorder

    print("Creating plot... (Strange interval)")
    h = Helicorder(st, interval=60, color=greyscale_hex,  # define the helicorder specs
                   show_y_UTC_label=True, right_vertical_labels=True,
                   title=st[0].id + ": Earthquakes at Goma",
                   )
    h.plot()  # plot the st
    h.plot_tags(UTCDateTime("2021/05/20 01:36:30"), color="yellow", markersize=8)  # plot a single time
    h.plot_tags(UTCDateTime("2021/05/20 17:19:35"), marker="|", markeredgecolor="blue", markersize=15)  # plot a single time as a P arrival
    h.plot_tags(UTCDateTime("2021/05/20 17:25:00"), marker="|", markeredgecolor="red", markersize=15)   # plot a single time as a S arrival
    h.plot_tags(UTCDateTime("2021/05/20 17:44:45"), marker="|", markeredgecolor="black", markersize=15)  # plot a single time as a coda end
    h.plot_tags([UTCDateTime("2021/05/20 08:31:00"), "2021/05/20 02:08:00"], color="yellow")  # plot a list of times given in any format

    plt.show()  # show the helicorder

    print("Done.")


if __name__ == "__main__":
    dev_xy_labels()
    main()
