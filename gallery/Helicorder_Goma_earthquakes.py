import numpy as np
import datetime as dt
from pytz import timezone
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
# from vdapseisutils import Helicorder
from vdapseisutils.core.swarmmpl.heli import Helicorder
from vdapseisutils.style.colors import greyscale_hex, swarm_colors_hex

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

    print("Creating default Helicorder...")
    h = Helicorder(st)
    plt.show()

    print("Creating Helicorder... (60 minute interval)")
    h = Helicorder(st, interval=60, color=swarm_colors_hex,  # define the helicorder specs
                   # show_y_UTC_label=True, right_vertical_labels=True,
                   title="Earthquakes in Goma (" + st[0].id + ")",
                   )
    h.plot_tags(UTCDateTime("2021/05/20 01:36:30"), color="yellow", markersize=8)  # plot a single time
    h.plot_tags(UTCDateTime("2021/05/20 17:19:35"), marker="|", markeredgecolor="blue", markersize=15)  # plot a single time as a P arrival
    h.plot_tags(UTCDateTime("2021/05/20 17:25:00"), marker="|", markeredgecolor="red", markersize=15)   # plot a single time as a S arrival
    h.plot_tags(UTCDateTime("2021/05/20 17:44:45"), marker="|", markeredgecolor="black", markersize=15)  # plot a single time as a coda end
    h.plot_tags([UTCDateTime("2021/05/20 08:31:00"), "2021/05/20 02:08:00"], color="yellow", marker="*", markersize=15)  # plot a list of times given in any format
    h.highlight([(UTCDateTime("2021/05/20 01:36"), UTCDateTime("2021/05/20 02:40"))])  # spans two lines (at 60')
    h.highlight([(UTCDateTime("2021/05/20 08:30"), UTCDateTime("2021/05/20 08:35"))], color="black")  # contained within 1 line (at 60')
    h.highlight([(UTCDateTime("2021/05/20 11:30"), UTCDateTime("2021/05/20 13:35"))], color="red")  # spans 3 lines (at 60')
    plt.show()  # show the helicorder

    print("Creating plot... (Strange interval)")
    h = Helicorder(st, interval=43, color=greyscale_hex,  # define the helicorder specs
                   )
    plt.show()  # show the helicorder

    print("Done.")


if __name__ == "__main__":
    main()
