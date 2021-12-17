from vdapseisutils.waveformutils.datasource import DataSource
from vdapseisutils.waveformutils.plotting.swarmmpl import swarmwg, swarmw


def main():
    ds = DataSource('ew', 'pubavo1.wr.usgs.gov:16022')
    nslc_list = ['AV.PV6A.--.SHZ', 'AV.PS4A.--.BHZ', 'AV.PVV.--.SHZ']  # Pavlof
    tstart = '2021/11/08 03:10'
    tend = '2021/11/08 03:50'
    st = ds.getWaveforms(nslc_list, tstart, tend, verbose=False)

    fig = swarmwg(st, wave_color='k', cmap='magma')
    fig = swarmw(st, color='w')


if __name__ == '__main__':
    main()
