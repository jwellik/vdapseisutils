import numpy as np
import obspy


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
        ds_string     : str : Path to files or string representation of ObsPy client

    EXAMPLES:
    # Many strings are approved if you want to use files
    >>> DataSource('file', '/path/to/top/directory', filepattern='*.mseed')
    >>> DataSource('files', '/path/to/top/directory', filepattern='*.mseed')
    >>> DataSource('filelist', '/path/to/top/directory', filepattern='*.mseed')
    >>> DataSource('filestructure', '/path/to/top/directory', filepattern='*.mseed')
    >>> DataSource('directory', '/path/to/top/directory', filepattern='*.mseed')

    # Various ObsPy client strings are supported
    # Server & Port are formatted as 'server:port'
    >>> DataSource('FDSN', 'IRIS')
    >>> DataSource('ew', '127.0.0.1:16022') # 'server:port'
    >>> DataSource('neic', '127.0.0.1:16022')

    """

    def __init__(self, ds_type, ds_string,
                 filepattern='*',  # Only used for Filestructure DataSource
                 timeout=60):  # Only used for Client DataSource

        # Filestructure DataSource
        if ds_type.lower() in ['file', 'files', 'filelist', 'filestructure', 'directory']:
            self.ds_type = 'filestructure'
            self.searchdir = ds_string  # top level directory for filestructure
            self.filepattern = filepattern  # regexp filepattern
            self.filelist_all = None  # Filelist
            self.filelist = None  # Filelist matching time and nslc
            self.name = '{} {}'.format(self.searchdir, self.filepattern)  # For printing purposes

        # Client DataSource
        elif self.ds_type.lower() in ['fdsn', 'earthworm', 'ew', 'wws', 'seedlink', 'slink']:
            self.ds_type = 'client'
            self.client_type = ds_type
            self.timeout = timeout
            self.client = self.create_client()
            self.name = None  # This is set later, when the client is created

        print('DataSource: {}'.format(self.name))

    def create_client(self):
        """Creates ObsPy Client based on the client type provided"""

        # DataSource is an FDSN server
        if self.ds_type.lower() in ['fdsn']:

            from obspy.clients.fdsn import Client
            self.name = 'FDSN Client {}'.format(self.ds_string)
            self.datasource = Client(self.ds_string)

        # DataSource is some other type of server
        else:

            server, port = self.ds_string.split(':');
            port = int(port)

            if self.ds_type.lower() in ['earthworm', 'ew', 'wws']:

                from obspy.clients.earthworm import Client
                self.name = 'EW Client {}:{}'.format(server, port)
                self.client = Client(server, port)

            elif self.ds_type.lower() in ['seedlink', 'slink']:

                from obspy.clients.seedlink import Client
                self.name = 'SeedLink Client {}:{}'.format(server, port)
                self.client = Client(server, port, timeout=self.timeout)


            elif self.ds_type.lower() in ['neic']:

                from obspy.clients.neic import Client
                self.name = 'NEIC Client {}:{}'.format(server, port)
                self.client = Client(server, port)

            else:

                print('>>> Client not supported')
                self.client = None
                self.client_type = '--'
                self.ds_type = '--'
                self.name = '--'

    def getWaveforms(self, nslc_list, tstart, tend, create_empty_trace=False, verbose=False):

        from vdapseisutils.waveformutils.datasource.fileutils import get_filelist, get_waveforms_from_file_sublist
        from vdapseisutils.waveformutils.datasource.clientutils import get_waveforms_from_client
        from obspy import Stream

        st = Stream()

        if self.ds_type.lower() == 'filestructure'.lower():

            if self.filelist is None:
                self.filelist = get_filelist(nslc_list, tstart, tend, filepattern=self.filepattern)

            st = get_waveforms_from_file_sublist(self.filelist_sub,
                                                      nslc_list, tstart, tend, create_empty_trace=create_empty_trace,
                                                      verbose=verbose)

        elif self.ds_type.lower() == 'client'.lower():

            st = get_waveforms_from_client(nslc_list, tstart, tend, create_empty_trace=create_empty_trace,
                                                verbose=verbose)

        return st

    # TO DO
    # [ ] Functionalize createFileList & createClient ??
    # [ ] Assume FDSN or EW or filelist, as appropriate if only one argument given???
    # [ ] Add functionality for other ObsPy client arguments

    """
    POSSIBLE DATASOURCE INITIALIZATIONS
    
    DataSource('IRIS')
    
    DataSource('vdap.org:16024')
    
    DataSource('/Users/jwellik/Data/sds/')
    
    DataSource('vdap.org:5002:/home/jwellik/data/sds/')
    
    DataSouce('pubavo.alaska.edu:18001')
    
    
    
    
    
    HOW TO HANDLE FILEPATTERN AND FILELIST STUFF
    
    ds = DataSource('files', '/Users/sysop/DATA/sds/', filepattern='*.mseed')
    > filelistall = create_filelist_all(searchdir, filepattern=filepattern)
    > filelist    = parse_filelist_for_nslc_time(filelist, nslc, t1, t2)
    
    
    ds.getWaveforms(nslc, t1, t2 )
    
    list_of_files = ds.getFileList( nslc, t1, t2 )
    > [filelist_all = get_filelist_all( searchdir, filepattern ) ] DONE WHEN DATASOURCE IS INITIALIZED!
    > list_of_files = parse_filelist_for_nslc_time( filelist_all, nslc, t1, t2 )
    
    
    OR
    
    list_of_files = ds.getFileList(nslc, t1, t2 )
    > filelist_all  = get_filelist_all( searchdir, filepattern )
    > list_of_files = parse_filelist_for_nslc_time( filelist_all, nslc, t1, t2 )
    
    OR
    
    list_of_files = ds.getFileList(nslc, t1, t2, filepattern )
    > filelist_all  = get_filelist_all(searchdir, filepattern)
     > list_of_files = parsefilelist_for_nslc_time(filelist_all, nslc, t2, t2 )
    """
