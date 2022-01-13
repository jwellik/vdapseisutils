import os

from vdapseisutils.waveformutils.datasource.fileutils import get_all_files, get_filelist


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
    >>> DataSource(['/path/to/file1', '/path/to/file2']

    # Various ObsPy client strings are supported
    # Server & Port are formatted as 'server:port'
    >>> DataSource('IRIS')  # FDSN severs are automaticlly recognized
    >>> DataSource('fdsnws://IRIS')
    >>> DataSource('127.0.0.1:16022') # Winston and Earthworm waveservers are automatically recognized
    >>> DataSource('waveserver://127.0.0.1:16022') # 'server:port'
    >>> DataSource('neic://127.0.0.1:16022')

    """

    def __init__(self, ds_input,
                 filepattern='*',  # Only used for Filestructure DataSource
                 timeout=60):  # Only used for Client DataSource

        # FileList DataSource (List of files)
        if type(ds_input) is list:
            print("DataSource is a list of files.")
            self.ds_type = 'filelist'
            self.searchdir = None
            self.filepattern = filepattern
            self.filelist_all = ds_input
            self.filelist = ds_input
            self.name = "(List of files) [{} ... {}]".format(self.filelist[0], self.filelist[-1])

        # Filestructure DataSource (path directory)
        elif os.path.isdir(ds_input):
            # if arginput.lower() in ['file', 'files', 'filelist', 'filestructure', 'directory']:
            print("DataSource is a filestructure.")
            self.ds_type = 'filestructure'
            self.searchdir = ds_input  # top level directory for filestructure
            self.filepattern = filepattern  # regexp filepattern
            self.filelist_all = get_all_files(ds_input, filepattern=self.filepattern)
            self.filelist = get_all_files(ds_input, filepattern=self.filepattern)
            self.name = '{} {}'.format(self.searchdir, self.filepattern)  # For printing purposes

        # Client DataSource
        else:
            # elif ds_input.lower() in ['fdsn', 'earthworm', 'ew', 'wws', 'seedlink', 'slink']:
            print("DataSource is an ObsPy Client")
            self.ds_type = 'client'
            self.client_type = None
            self.timeout = timeout
            self.name = None  # This is set later, when the client is created
            self.client = None  # This is set later, when the client is created
            self.create_client(ds_input)  # creates self.client & self.name
            print(self.client)

        print('DataSource: {}'.format(self.name))

    def create_client(self, client_str):
        """Creates ObsPy Client based on the client type provided"""

        # 1) Determine client type
        # 2) Make Client

        # Backward compatibility -- Assign server type is not provided
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

        elif 'waveserver://' in client_str:
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

            st = get_waveforms_from_client(self.client, nslc_list, tstart, tend, create_empty_trace=create_empty_trace,
                                           verbose=verbose)

        return st

    def archive(self):
        print(">>> Currently not yet implemented!")
        pass

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
