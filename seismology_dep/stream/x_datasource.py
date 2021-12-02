# Datasource V1

import obspy


def datasource( datasource, verbose=False ):
    "Returns a list of files to read or a waveserver client"
    
    # Generate list of files to load from file structure
    if 'file:' in datasource:
        if verbose: print('Generating list of files from {}'.format(datasource))
        ds = createFileList( datasource )
        
    elif 'IRIS' in datasource:
        from obspy.clients.fdsn.client import Client
        ds = Client('IRIS')
    
    # Generate Waveserver client
    else:
        if verbose: print('Generating server client {}'.format(datasource))
        ds = createClient( datasource )   
    
    return ds



def createClient( ds, verbose=False ):
    """
    CREATECLIENT Creates an ObsPy client from a datasource string

    # read data from Earthworm
    >>> createclient( 'ew:192.1.0.1:16000')

    # read data from Winston
    >>> createclient( 'wws:192.1.0.1:16000')

    # FDSNW
    >>> createclient( 'fdsnw:192.1.0.1:16000')

    # SeedLink
    >>> createclient( 'seedlink:192.1.0.1:16000')


    """

    # Earthworm, Winston, or other Waveserver
    #if ['ew:', 'wws:', 'waveserver:'] in ds:
    if any([substring in ds for substring in ['ew:', 'wws:', 'waveserver:'] ]):
        from obspy.clients.earthworm import Client as EWClient
        server, port = ds.split(':')[1:]
        client = EWClient(server, int(port))
        
    # New server syntax (more options and server and port on same variable)
    elif 'fdsnws:' in ds:
        from obspy.clients.fdsn import Client
        server = ds.split(':')[1]
        client = Client(server)
        
    elif 'seedlink:' in ds:
        from obspy.clients.seedlink import Client as SeedLinkClient
        server, port = ds.split(':')[1:]
        client = SeedLinkClient(server, int(port), timeout=1)
        
    else:
        print('Failed to create client from {}'.format(ds))
        
    return client
            


def createFileList( ds, verbose=False ):
# Usage

# read data from file structure
# automatically searches all subdirectories
# Directory must end with '/' or equivalent symbol for your operating system
#>>> createclient( 'file:/home/seismic/data/AV/' ) # specifiy only a directory
#
#>>> createclient( 'file:/home/seismic/data/AV/*mseed' ) # specift directory and search pattern
    
    ds=ds.split(':')[1:][0] # 'file:/home/data/' --> '/home/data/' (gets rid of 'file:')

    # Generate list of files
    searchdir, filepattern = os.path.split(ds)
    flist = list(itertools.chain.from_iterable(glob.iglob(os.path.join(
            root,filepattern)) for root, dirs, files in os.walk(searchdir)))

    # Determine which subset of files to load based on start and end times and
    # station name; we'll fully deal with stations below
    flist_sub = []
    for f in flist:
        # Load header only
        stmp = obspy.read(f, headonly=True)
        # Check if station is contained in the stas list
        if stmp[0].stats.station in stas:
            # Check if contains either start or end time
            ststart = stmp[0].stats.starttime
            stend = stmp[-1].stats.endtime
            if (ststart<=tstart and tstart<=stend) or (ststart<=tend and
                tend<=stend) or (tstart<=stend and ststart<=tend):
                flist_sub.append(f)


#        # Fully load data from file
#        stmp = Stream()
#        for f in flist_sub:
#            tmp = obspy.read(f, starttime=tstart, endtime=tend+opt.maxdt)
#            if len(tmp) > 0:
#                stmp = stmp.extend(tmp)

    return flist_sub


def get_waveforms_from_server(
            client,                # obspy Client
            nslc_list,
            tstart,
            tend,
            filelength=86400,
            max_download=3600,
            clean=True,
            fill_value=None,
            filterargs=[],
            taperargs=[],
            verbose=False
    ):
    
    
    import timeutils
    
    from obspy import UTCDateTime, Stream

    import seismology.stream
    from seismology.stream.nslcobject import str2nslc, setNSLC
    from seismology.stream import removeWinstonGaps

    from ioutils.filestructure import write2sds
    
    client = client
    
    # Assert tstart, tend as UTCDateTime
    tstart = UTCDateTime(tstart)
    tend   = UTCDateTime(tend)

    
    # Assert proper NSLC lists
    if type(nslc_list) is str: nslc_list = [nslc_list]   # Assert NSLC as list
    if nslc_out_list is None: nslc_out_list = nslc_list  # Assert that nslc_list_exists
    if len(nslc_list)!=len(nslc_out_list): print('Warning: Lengths of NLSC input and output lists do not match!')

    # Loop through list of NSLCs
    for nslc, nslc_out in zip(nslc_list, nslc_out_list):

        print('- Loading {}'.format(nslc))
        net, sta, loc, cha = str2nslc( nslc )


        st = Stream()

        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.createTimeChunks(tstart, tend, nsec=max_download, verbose=False)
        for dt1, dt2 in zip(dtstarts, dtends):
            if verbose: print('  - Downloading   : {} to {}'.format(dt1, dt2))

            try:
                stmp = client.get_waveforms(net, sta, loc, cha, dt1, dt2)
                stmp = removeWinstonGaps(stmp) # Loops through Streams

                # Apply standard clean
                if clean:
                    if filterargs:
                        stmp = stmp.filter(filterargs)
                    else:
                        stmp = stmp.detrend('demean')
                    stmp = stmp.taper(max_percentage=0.01)

                for tr in stmp: # Deal w error when sub-traces have different dtypes
                    if tr.data.dtype.name != 'int32': tr.data=tr.data.astype('int32') # force type int32
                    if tr.data.dtype!=dtype('int32'): tr.data=tr.data.astype('int32') # force type int32
                # deal with rare error when sub-traces have different sample rates    

                stmp.merge(fill_value=fill_value)

            except:
                stmp = Stream()

            # Create empty trace
            if not stmp:
                from obspy import Trace

                stmp=Trace()
                stmp.stats['network']  = net
                stmp.stats['station']  = sta
                stmp.stats['location'] = loc
                stmp.stats['channel']  = cha
                stmp.stats['sampling_rate'] = 100 
                stmp.stats['starttime']     = dt1
                stmp.data=zeros(int((dt2-dt1)*tr.stats['sampling_rate']),dtype='int32')

            st += stmp

        st = st.merge(method=1, fill_value=fill_value)
        st = st.slice(pt1, pt2, nearest_sample=False) # Ensures that an extra sample is not included at the end (Or use trim?)
        st = setNSLC( st, nslc_out)

    return st


def get_waveforms_from_file():
    pass


def get_clean_waveforms_from_file():
    pass