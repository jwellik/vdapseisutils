import os


class DataSource:
    """DATASOURCE Create ObsPy Client or list of files that can retrieve waveforms the same way

    ARGUMENTS:
        ds_type       : str : The type of DataSource (either files or ObsPy client)
                                'filestructure', aka --> 'file', 'files', 'filelist', 'directory'

                                Client needs to be defined as -->
                                   FDSN      --> 'fdsn'
                                   Earthworm --> 'earthworm', 'ew', 'wws'
                                   SeedLink  --> 'seedlink', 'slink'
                                   NEIC      --> 'neic'
        ds_input     : str : Path to files or string representation of ObsPy client

    EXAMPLES:
    # Many strings are approved if you want to use files
    >>> DataSource('/path/to/top/directory')
    >>> DataSource(['/path/to/file1', '/path/to/file2'])

    # Various ObsPy client strings are supported
    # Server & Port are formatted as 'server:port'
    >>> DataSource('IRIS')  # FDSN severs are automatically recognized
    >>> DataSource('fdsnws://IRIS')
    >>> DataSource('127.0.0.1:16022') # Winston and Earthworm waveservers are automatically recognized
    >>> DataSource('waveserver://127.0.0.1:16022') # 'server:port'
    >>> DataSource('neic://127.0.0.1:16022')

    """

    # TODO Remove filestructure compatability. Force user to use SDS filesystem.

    def __init__(self, ds_input,
                 filepattern='*',  # Only used for Filestructure DataSource
                 timeout=60,       # Only used for Client DataSource
                 verbose=False,
                 ):

        # DataSource are files
        if (type(ds_input)) is list or (os.path.isdir(ds_input)):
            raise Exception("Sorry. Files are not supported at this time (SDS is available as a Client option.")

        # DataSource is an ObsPy Client
        else:
            self.ds_type = 'client'
            # self.client_type = None
            self.timeout = timeout
            self.name = None  # This is set later, when the client is created
            self.client = None  # This is set later, when the client is created
            self.__create_client(ds_input)  # creates self.client & self.name
            # print(self.client)

        if verbose:
            self.print()

    def print(self):
        print("DataSource ({}) : {}".format(self.ds_type, self.name))

    def __create_client(self, client_str):
        """Creates ObsPy Client based on the client type provided"""

        # 1) Determine client type
        # 2) Make Client

        # Backward compatibility -- Assign server type if not provided
        if '://' not in client_str:
            if '.' not in client_str:  # e.g., DataSource('IRIS')
                client_str = 'fdsnws://' + client_str
            else:  # e.g., DataSource("vdap.org:1600")
                client_str = 'waveserver://' + client_str

        # New server syntax (more options and server and port on same variable)
        if 'fdsnws://' in client_str:
            from obspy.clients.fdsn import Client
            server = client_str.split('fdsnws://', 1)[1]
            self.name = 'FDSN Client {}'.format(client_str)
            self.client = Client(server)

        # ObsPy Earthworm Client is interpretted from "earthworm", "waveserver", "winston", or "wws"
        elif any(string in client_str for string in ["waveserver://", "earthworm://", "winston://", "wws://"]):
            from obspy.clients.earthworm import Client
            serverport = client_str.split('waveserver://', 1)[1]
            server, port = serverport.split(':')
            port = int(port)
            self.name = 'Waveserver Client {}:{}'.format(server, port)
            self.client = Client(server, port)

        elif 'seedlink://' in client_str:
            from obspy.clients.seedlink import Client
            serverport = client_str.split('seedlink://', 1)[1]
            server, port = serverport.split(':')
            self.name = 'SeedLink Client {}:{}'.format(server, port)
            self.client = Client(server, port, timeout=self.timeout)

        elif 'neic://' in client_str:
            from obspy.clients.neic import Client
            serverport = client_str.split('neic://', 1)[1]
            server, port = serverport.split(':')
            self.name = 'NEIC Client {}:{}'.format(server, port)
            self.client = Client(server, port)

        elif 'sds://' in client_str:
            from obspy.clients.filesystem.sds import Client
            sdspath = client_str.split('sds://', 1)[1]
            self.name = 'SDS {}'.format(sdspath)
            self.client = Client(sdspath)

        else:

            print('>>> Client not supported')
            self.client = None
            # self.client_type = '--'
            self.ds_type = '--'
            self.name = '--'


    def getWaveforms(self, nslc_list, tstart, tend, max_download="1D", create_empty_trace=False, fill_value=None, verbose=False):

        import pandas as pd
        from obspy import Stream, UTCDateTime
        from vdapseisutils.core.datasource.clientutils import get_waveforms_from_client

        st = Stream()
        tstart = UTCDateTime(tstart)
        tend = UTCDateTime(tend) - pd.Timedelta(seconds=0.0001)  # Remove 0.0001 sample to avoid surplus sample point

        if self.ds_type.lower() == 'client'.lower():

            if verbose:
                print("Downloading miniseed data...")
            st = get_waveforms_from_client(self.client, nslc_list, tstart, tend, max_download=max_download,
                                           create_empty_trace=create_empty_trace,
                                           fill_value=fill_value, verbose=verbose)

            if verbose:
                print("Done.")

        return st

    def archive_waveforms(self, nslc_list, tstart, tend, loc="--", resample=False,
                          basedir="./", mplex_archive=False, verbose=False):
        """archive_waveforms Archives waveforms as miniseed files with reclen 512 in the SDS filestructure"""

        import pandas as pd
        from obspy import Stream, UTCDateTime
        from vdapseisutils.core.datasource.clientutils import get_waveforms_from_client
        from vdapseisutils.utils import timeutils

        if verbose:
            print("Archiving as miniseed files...")

        max_download = "1D"
        create_empty_trace = False
        fill_value = None

        st = Stream()
        tstart = UTCDateTime(tstart)
        tend = UTCDateTime(tend) #- pd.Timedelta(seconds=0.0001)  # Remove 0.0001 sample to avoid surplus sample point


        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.time_range(tstart, tend, freq=max_download)
        for dt1, dt2 in zip(dtstarts, dtends):

            dt2 = UTCDateTime(UTCDateTime(dt2) - pd.Timedelta(seconds=0.0001))  # Remove 0.0001 sample to avoid surplus sample point
            st_day = Stream()

            for nslc in nslc_list:

                st = get_waveforms_from_client(self.client, nslc, dt1, dt2, max_download=max_download,
                                               create_empty_trace=create_empty_trace,
                                               fill_value=fill_value, verbose=verbose)
                if resample:
                    st.resample(sampling_rate=resample)
                st_day += st
                st.merge()  # Merges Streams (combines like NSLCs); creates Masked array

                from pathlib import Path
                basedir = Path(basedir)
                for tr in st:

                    tr.stats.location = loc

                    tmp_st = tr.split()  # Splits Traces with masked array into Stream w multiple Traces and no masked arrays; can be written

                    try:
                        sdspath = "{yyyy}/{net}/{sta}/{chan}".format(yyyy=tr.stats.starttime.year, net=tr.stats.network,
                                                                     sta=tr.stats.station, chan=tr.stats.channel)
                        sdspath = Path(os.path.join(basedir, sdspath))
                        sdspath.mkdir(parents=True, exist_ok=True)
                        sdsname = "{id}.D.{yyyy}.{jday:03}.mseed".format(yyyy=dt1.year, id=tr.id,
                                                                    jday=dt1.julday)
                        # sdsname = "{id}.D.{yyyy}.{jday:03}.mseed".format(yyyy=tr.stats.starttime.year, id=tr.id,
                        #                                             jday=tr.stats.starttime.julday)
                        fullpath = os.path.join(basedir, sdspath, sdsname)
                        tmp_st.write(fullpath, reclen=512)
                        if verbose:
                            print("  -> SUCCESS : {}".format(fullpath))
                    except:
                        if verbose:
                            print("  -> Failed! : {}".format(fullpath))

            if mplex_archive:
                mplex_archive = Path(mplex_archive)
                mplex_archive.mkdir(parents=True, exist_ok=True)
                filename = "{yyyy:04d}-{mm:02d}-{dd:02d}.mseed".format(yyyy=dt1.year,
                                                           mm=dt1.month,
                                                           dd=dt1.day)
                fullpath = os.path.join(mplex_archive, filename)
                st_day.write(fullpath)
                if verbose:
                    print("Multiplexed Miniseed File Written: {}".format(fullpath))

        if verbose:
            print("Done.")

