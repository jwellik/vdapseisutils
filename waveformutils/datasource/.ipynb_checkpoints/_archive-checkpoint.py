from waveformutils.ioutils.filestructure import sds_standard

def archive( self, nslc_list, tstart, tend,
            basedir='./',
            filestructure=sds_standard,
            reclen=4096,
            filelength=86400,
            max_download=3600,
            return_stream=False,
            verbose=False,
           ):
    
    """ARCHIVE Archives Waveserver data as miniseed files in a SDS file structure.
        Does not clean data but will fill Winston Gaps w 0.
    
    Input arguments:
    
        ds                     : DataSourse : 
        nslc_list              : list or str : list of NSLC strings (or just one string)
        tstart                 : UTCDateTime or date format understood by UTCDateTime
        tend                   : UTCDateTime or date format understood by UTCDateTime
        basedir                : str : filepath for base directory of SDS archive
        reclen                 : int : Should be set to the desired data record length in bytes
                                       which must be expressible as 2 raised to the power of X
                                       where X is between (and including) 8 to 20
                                       default : 4096, usually for archiving
                                       512 byte record length, usually for streaming & playback
                                       https://docs.obspy.org/packages/autogen/obspy.io.mseed.core._write_mseed.html#obspy.io.mseed.core._write_mseed
        filelength             : float : length of each file in seconds
        max_download           : float : maximum number of seconds to download at a time
        vebose                 : bool
        
    
    """
    
    from obspy import UTCDateTime, Stream

    import waveformutils.datasource
    from waveformutils.nslcutils import str2nslc, setNSLC
    from waveformutils import nslcutils
    from waveformutils.streamutils import removeWinstonGaps
    from waveformutils import timeutils as timeutils
    from waveformutils import ioutils as ioutils
    from waveformutils.ioutils.filestructure import write2sds
    
    st_final = Stream()
    
    # Establish datasource
    print('Archiving data from : {}'.format(self.name))
    print('Archive destination : <{}>{}'.format(basedir, filestructure))
    client = self.datasource

    # Assert NSLC as list
    if type(nslc_list) is str: nslc_list = [nslc_list]

    # Assert tstart, tend as UTCDateTime
    tstart = UTCDateTime(tstart)
    tend   = UTCDateTime(tend)

    # Assert nslc_out_list
    #if nslc_out_list is None: nslc_out_list = nslc_list

    # Loop through list of NSLCs
    #for nslc, nslc_out in zip(nslc_list, nslc_out_list):
    for nslc in nslc_list:
    
        #print('- Accessing {} as {}'.format(nslc, nslc_out))
        print('- Accessing {}'.format(nslc))
        net, sta, loc, cha = str2nslc( nslc )

        # Loop through time chunks for output file
        proctstarts, proctends = timeutils.createTimeChunks(tstart, tend, nsec=filelength, verbose=False)
        for pt1, pt2 in zip(proctstarts, proctends):
            print(' - Processing    : {} to {}'.format(pt1, pt2))

            st = Stream()

            # Create smaller time chunks to download, if necessary
            dtstarts, dtends = timeutils.createTimeChunks(pt1, pt2, nsec=max_download, verbose=False)
            for dt1, dt2 in zip(dtstarts, dtends):
                if verbose: print('  - Downloading   : {} to {}'.format(dt1, dt2))

                try:
                    stmp = client.get_waveforms(net, sta, loc, cha, dt1, dt2)
                    #stmp = removeWinstonGaps(stmp) # Loops through Streams

                except:
                    stmp = Stream()

                st += stmp

            st = st.merge(method=1, fill_value=0)
            st = st.slice(pt1, pt2, nearest_sample=False) # Ensures that an extra sample is not included at the end (Not sure this is doing anything)
            #st = setNSLC( st, nslc_out)
            for tr in st:
                print('   - Retrieved   : {} | {} to {} | {} Hz, {} samples'.format(tr.id, tr.stats.starttime, tr.stats.endtime, tr.stats.sampling_rate, tr.stats.npts))

            # Write Stream to File
            if len(st)>0:
                filenames = write2sds(st, basedir=basedir, filestructure=filestructure, fileformat='mseed', reclen=reclen)
                for f in filenames:
                    print(' - Archived! >>> {}'.format(f))
                #print(' - Archived      : {}'.format(f))
                #print('                   {} to {}'.format(st[0].stats.starttime, st[0].stats.endtime))
                
                if return_stream:
                    st_final += st
                
            else:
                print(' - No Streams returned : {} ({} to {})'.format(nslc, pt1, pt2))
                
    print('Done.')
    #return st_final