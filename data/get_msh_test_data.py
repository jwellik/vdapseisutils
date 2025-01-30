from obspy import UTCDateTime
from obspy.geodetics.base import kilometers2degrees as km2d
from obspy.core.event.base import WaveformStreamID as ObsPyStreamID
from obspy.clients.fdsn import Client
# from vdapseisutils.core.datasource.nslcutils import WaveformStreamID
from vdapseisutils.utils.obspyutils.inventoryutils import inventory2df, df2inventory, overwrite_loc_code


def main():

    t1 = UTCDateTime("2024/07/01")
    t2 = UTCDateTime("2024/07/01 23:59:59.9999")

    # # Station List - Everything within 50k from Mt. Hood (as recommended by Wes)
    # nslc_list = [
    #     "CC.BRSP..BHZ", "CC.HIYU..BHZ", "CC.LSON..BHZ", "CC.PALM..BHZ", "CC.SHRK..BHZ", "CC.TIMB..BHZ", "CC.YOCR..BHZ",
    #     "UW.HOOD..ENZ", "UW.HOOD..HHZ", "UW.VLL..EHZ", "UW.THD.."]
    # bulk = [(*nslc.split("."), t1, t2) for nslc in nslc_list]

    client = Client("IRIS")
    # inv = client.get_stations_bulk(bulk)
    # print(inv)

    print("Gettings stations...")
    inv2 = client.get_stations(latitude=46.2002, longitude=-122.1855, maxradius=km2d(5.0),
                               network="CC", channel="BHZ", level="channel",
                               starttime=UTCDateTime(t1), endtime=UTCDateTime(t2))
    print(inv2)
    df = inventory2df(inv2)
    df = df.drop_duplicates(subset=["nslc"])
    inv2.write("/home/jwellik/Downloads/msh_inventory.xml", format="STATIONXML")

    print("Getting waveforms...")
    bulk = [(*nslc.split("."), t1, t2) for nslc in df["nslc"]]
    st = client.get_waveforms_bulk(bulk)
    st.write("msh_waveforms.mseed", format="MSEED")

    print("Done.")


if __name__ == "__main__":
    main()
