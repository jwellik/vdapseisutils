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

    client = Client("IRIS")
    print("Getting waveforms...")
    st = client.get_waveforms("AV", "GALA", "--", "BHZ", start, stop)
    st.write("gareloi_waveform.mseed")

    print("Creating Helicorder... (60 minute interval)")
    h = Helicorder(st, interval=60, color=greyscale_hex,  # define the helicorder specs
                   title="Gareloi, Alaska (" + st[0].id + ")",
                   )
    h.highlight([(UTCDateTime("2021/05/20 03:03"), UTCDateTime("2021/05/20 03:08"))], color="blue")
    h.highlight([(UTCDateTime("2021/05/20 19:16"), UTCDateTime("2021/05/20 19:19"))], color="red")  # contained within 1 line (at 60')
    h.highlight([(UTCDateTime("2021/05/20 19:23"), UTCDateTime("2021/05/20 19:27"))], color="blue")  # spans 3 lines (at 60')
    plt.show()  # show the helicorder

    print("Done.")


if __name__ == "__main__":
    main()
