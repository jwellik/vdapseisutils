import numpy as np
import obspy


class DataSource:
    """DATASOURCE Create ObsPy Client or list of files that can retrieve waveforms the same way
    
    ARGUMENTS:
        ds_type       : str : The type of DataSource (either files or ObsPy client)
                                Filelist  --> 'file', 'files', 'filelist', 'filestructure', 'directory'
                                FDSN      --> 'fdsn'
                                Earthworm --> 'earthworm', 'ew', 'wws'
                                SeedLink  --> 'seedlink', 'slink'
                                NEIC      --> 'neic'
        ds_string     : str : Path to files or string representation of ObsPy client
        
    EXAMPLES:
    # Many strings are approved if you want to use files
    >>> DataSource('file', '/path/to/top/directory')
    >>> DataSource('files', '/path/to/top/directory')
    >>> DataSource('filelist', '/path/to/top/directory')
    >>> DataSource('filestructure', '/path/to/top/directory')
    >>> DataSource('directory', '/path/to/top/directory')
    
    # Various ObsPy client strings are supported
    # Server & Port are formatted as 'server:port'
    >>> DataSource('FDSN', 'IRIS')
    >>> DataSource('ew', '127.0.0.1:16022') # 'server:port'
    >>> DataSource('neic', '127.0.0.1:16022')
    
    """

    from ._get_waveforms import get_waveforms_from_files, get_waveforms_from_client
    from ._archive import archive

    def __init__(self, ds_type, ds_string, filepattern='*'):

        import glob, os, itertools

        self.name = ds_string  # string version of datasource; original user input
        self.ds_type = ds_type  # filelist or client type
        self.datasource = []  # ObsPy Client or a list of files (created by the class)

        # if datasource is a filestructure >>> createFileList()
        if self.ds_type.lower() in ['file', 'files', 'filelist', 'filestructure', 'directory']:
            if os.path.isdir(ds_string):
                # create file list
                flist = list(itertools.chain.from_iterable(glob.iglob(os.path.join(
                    root, filepattern)) for root, dirs, files in os.walk(ds_string)))

                self.ds_type = 'filelist'
                self.datasource = flist


        # else datasource is a server/Client >>> createClient()
        elif self.ds_type.lower() in ['fdsn', 'earthworm', 'ew', 'wws', 'seedlink', 'slink']:

            # DataSource is an FDSN server >>> createFDSNClient()
            if self.ds_type.lower() in ['fdsn']:

                from obspy.clients.fdsn import Client
                self.name = 'FDSN Client {}'.format(ds_string)
                self.ds_type = 'client'
                self.datasource = Client(ds_string)

            # DataSource is some other type of server
            else:

                server, port = ds_string.split(':');
                port = int(port)

                if self.ds_type.lower() in ['earthworm', 'ew', 'wws']:

                    from obspy.clients.earthworm import Client
                    self.name = 'EW Client {}:{}'.format(server, port)
                    self.ds_type = 'client'
                    self.datasource = Client(server, port)

                elif self.ds_type.lower() in ['seedlink', 'slink']:

                    from obspy.clients.seedlink import Client
                    self.name = 'SeedLink Client {}:{}'.format(server, port)
                    self.ds_type = 'client'
                    self.datasource = Client(server, port, timeout=1)


                elif self.ds_type.lower() in ['neic']:

                    from obspy.clients.neic import Client
                    self.name = 'NEIC Client {}:{}'.format(server, port)
                    self.ds_type = 'client'
                    self.datasource = Client(host=server, port=port, timeout=30)

                else:

                    print('>>> Client not supported')
                    self.datasource = ''
                    self.name = ''
                    self.type = ''
                    self.client_type = ''

        # print('-'*60)
        print('DataSource: {}'.format(self.name))
        # print('-'*60)

    def getWaveforms(self, nslc_list, tstart, tend,
                     create_empty_trace=False,
                     verbose=False):

        from obspy import Stream

        st = Stream()

        if self.ds_type.lower() == 'filelist'.lower():

            st = self.get_waveforms_from_files(nslc_list, tstart, tend, create_empty_trace=create_empty_trace,
                                               verbose=verbose)

        elif self.ds_type.lower() == 'client'.lower():

            st = self.get_waveforms_from_client(nslc_list, tstart, tend, create_empty_trace=create_empty_trace,
                                                verbose=verbose)

        return st

    def archive():
        return []

    def getFileList():
        return []

    def getDataGaps():
        print('Only available for Winston Wave Servers')
        return []

# TO DO
# [ ] Functionalize createFileList & createClient ??
# [ ] Assume FDSN or EW or filelist, as appropriate if only one argument given???
# [ ] Add functionality for other ObsPy client arguments
