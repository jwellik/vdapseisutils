

def main():

    from vdapseisutils.waveformutils.DataSource import DataSource

    print(':'*80)
    ds = DataSource("IRIS")
    st = ds.getWaveforms(['UW.JUN.--.EHZ'], '2004/10/15', '2004/10/15 04:00:00', create_empty_trace=True)
    st.plot()
    print(st)

    print(':'*80)
    ds = DataSource("pubavo1.wr.usgs.gov:16023")
    st = ds.getWaveforms(["AV.GAEA.--.BHZ"], "2022/01/13 01:41", "2022/01/13 01:50")
    st.plot()

    print(':'*80)
    ds = DataSource("waveserver://pubavo1.wr.usgs.gov:16023")

    print(':'*80)
    ds = DataSource("/Users/jwellik/Dropbox/JAY-DATA/TEST_DATA", filepattern=".mseed")

    print(':'*80)
    filelist = [
        "/Users/jwellik/Dropbox/JAY-DATA/TEST_DATA/2020-03-31-ml68-western-idaho/2020-03-31-ml68-western-idaho-mseed.miniseed",
        "/Users/jwellik/Dropbox/JAY-DATA/TEST_DATA/Test_Files/SRBI_APR_2017_Package_1506664425289-BMKG_669224_BMKG.mseed"
    ]
    ds = DataSource(filelist)


if __name__ == '__main__':
    main()