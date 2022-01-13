

def main():

    from waveformutils.DataSource import DataSource

    print(':'*80)
    ds = DataSource("IRIS")

    print(':'*80)
    ds = DataSource("vdap.org:16024")

    print(':'*80)
    ds = DataSource("waveserver://vdap.org:16024")

    print(':'*80)
    ds = DataSource("/Users/jwellik/Dropbox/JAY-DATA/TEST_DATA", filepattern=".mseed")

    print(':'*80)
    filelist = [
        "/Users/jwellik/Dropbox/JAY-DATA/TEST_DATA/2020-03-31-ml68-western-idaho/2020-03-31-ml68-western-idaho-mseed.miniseed",
        "/ Users / jwellik / Dropbox / JAY - DATA / TEST_DATA / Test_Files / SRBI_APR_2017_Package_1506664425289 - BMKG_669224_BMKG.mseed"
    ]
    ds = DataSource(filelist)


if __name__ == '__main__':
    main()