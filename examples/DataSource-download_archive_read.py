from waveformutils.datasource import DataSource
from obspy import UTCDateTime

def main():

    t1 = UTCDateTime('2004/10/15')
    t2 = UTCDateTime('2004/10/16')
    ds = DataSource('FDSN', 'IRIS')
    st = ds.getWaveforms(['UW.SEP..EHZ', 'UW.SHW..EHZ'], t1, t2, create_empty_trace=True)
    print(st)



if __name__ == '__main__':
    main()
