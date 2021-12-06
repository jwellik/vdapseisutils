import obspy
import numpy as np


def get_filelist(self, nslc_list, tstart, tend, output=None):
    '''GET_FILELIST

    If 'output' is specified, it will overwrite the filesublist to the specified file
    '''

    from obspy import UTCDateTime
    from waveformutils.nslcutils import getNSLCstr

    print('>>> waveforms.datasource[__init__].get_filelist()')

    tstart = UTCDateTime(tstart)
    tend = UTCDateTime(tend)

    # Create file sublist (only files w relevant nslc & time)
    # Determine which subset of files to load based on start and end times and
    # station name; we'll fully deal with stations below
    flist = self.datasource
    flist_sub = []
    print(flist)
    for f in flist:
        # Load header only
        stmp = obspy.read(f, headonly=True)
        # Check if station is contained in the stas list
        if getNSLCstr(stmp[0]) in nslc_list:  # if stmp[0].stats.station in stas:
            # Check if contains either start or end time
            ststart = stmp[0].stats.starttime
            stend = stmp[-1].stats.endtime
            if (ststart <= tstart <= stend) or (ststart <= tend <= stend) or (tstart <= stend and ststart <= tend):
                flist_sub.append(f)
    print(flist_sub)

    if output:
        textfile = open(output, "w+")
        for element in flist_sub:
            textfile.write(element + "\n")
        textfile.close()
        pass

    return flist_sub


def get_waveforms_from_files(self, nslc_list, tstart, tend,
                             fill_value=None,
                             create_empty_trace=False,
                             verbose=False):
    from obspy import UTCDateTime, Stream, Trace
    from waveformutils.nslcutils import getNSLCstr
    from waveformutils.streamutils import sortStreamByNSLClist

    print('>>> waveforms.datasource[__init__].get_waveforms_from_files()')

    tstart = UTCDateTime(tstart)
    tend = UTCDateTime(tend)

    flist_sub = get_filelist(self, nslc_list, tstart, tend)

    # Fully load data from file
    stmp = Stream()
    for f in flist_sub:
        tmp = obspy.read(f, starttime=tstart, endtime=tend)
        if len(tmp) > 0:
            stmp = stmp.extend(tmp)

    #         # Filter and merge
    #         stmp = stmp.filter('bandpass', freqmin=opt.fmin, freqmax=opt.fmax, corners=2,
    #             zerophase=True)
    #         stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
    #         for m in range(len(stmp)):
    #             if stmp[m].stats.sampling_rate != opt.samprate:
    #                 stmp[m] = stmp[m].resample(opt.samprate)
    #         stmp = stmp.merge(method=1, fill_value=0)

    stmp = stmp.merge(method=1, fill_value=fill_value)

    #     # Only grab stations/channels that we want and in order - Modelled after Alicia's code in REDPy
    #     # But this already happens?
    #     st = Stream()
    #     nslc_loaded = []
    #     for s in stmp:
    #         nslc_loaded.append(getNSLCstr(s))
    #     print(nslc_loaded)

    #     # Only grab stations in our nslc_list
    #     for n in range(len(nslc_list)):
    #         for m in range(len(nslc_loaded)):
    #             print('{} == {} ? --> '.format(nslc_list[n], nslc_loaded[m], nslc_list[n] in nslc_loaded[m]))
    #             if nslc_list[n] in nslc_loaded[m]:
    #                 st = st.append(stmp[m])
    #         print('len(st) {} == n {} == m {}?'.format(len(st),n,m))
    #         if len(st) == n:
    #             print('Could not find file for {}'.format(nslc_list[n]))
    #             trtmp = create_empty_trace(nslc_list[m], tstart, tend)

    if create_empty_trace == True:
        nslc_loaded = []
        for tr in stmp:
            nslc_loaded.append(getNSLCstr(tr))

        emptytr = Trace()
        for nslc in nslc_list:
            if nslc not in nslc_loaded:
                emptytr = createEmptyTrace(nslc, tstart, tend, sampling_rate=100)
        stmp += emptytr

    st = sortStreamByNSLClist(stmp, nslc_list)

    return st


def get_waveforms_from_client(
        self, nslc_list, tstart, tend,
        filelength=86400,
        max_download=3600,
        clean=False,
        fill_value=None,
        filterargs=[],
        taperargs=[],
        create_empty_trace=False,
        verbose=False
):
    print('>>> waveformutils.datasource[__init__].get_waveforms_from_client()')

    from obspy import UTCDateTime, Stream

    from waveformutils.nslcutils import str2nslc, setNSLC
    from waveformutils.streamutils import removeWinstonGaps, sortStreamByNSLClist
    from waveformutils import timeutils

    client = self.datasource

    # Assert tstart, tend as UTCDateTime
    tstart = UTCDateTime(tstart)
    tend = UTCDateTime(tend)

    # Assert proper NSLC lists
    if type(nslc_list) is str: nslc_list = [nslc_list]  # Assert NSLC as list

    st = Stream()

    # Loop through list of NSLCs
    for nslc in nslc_list:

        print('- Loading {}'.format(nslc))
        net, sta, loc, cha = str2nslc(nslc)

        stmp = Stream()

        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.createTimeChunks(tstart, tend, nsec=max_download, verbose=False)
        for dt1, dt2 in zip(dtstarts, dtends):

            stmp2 = Stream()

            if verbose: print('  - Downloading   : {} to {}'.format(dt1, dt2))

            try:
                #                 print('  - Getting waveform')
                stmp2 = client.get_waveforms(net, sta, loc, cha, dt1, dt2)  # Call ObsPy function
                #                 print(stmp2)
                # stmp = removeWinstonGaps(stmp) # Loops through Streams

                #                 # Apply standard clean
                #                 if clean:
                #                     if filterargs:
                #                         stmp = stmp.filter(filterargs)
                #                     else:
                #                         stmp = stmp.detrend('demean')
                #                     stmp = stmp.taper(max_percentage=0.01)

                #                     for tr in stmp: # Deal w error when sub-traces have different dtypes
                #                         if tr.data.dtype.name != 'int32': tr.data=tr.data.astype('int32') # force type int32
                #                         if tr.data.dtype!=dtype('int32'): tr.data=tr.data.astype('int32') # force type int32
                #                     # deal with rare error when sub-traces have different sample rates
                #                 print('  - Merging stream')
                stmp2.merge(fill_value=fill_value)
            #                 print(stmp2)

            except:
                # print('  - Unable to load data.')
                stmp2 = Stream()

            # Now operating on *this* download segment from *this* NSLC
            #             print('  - Appending & merging stream')
            stmp += stmp2
            stmp = stmp.merge(method=1, fill_value=fill_value)
        #             print(stmp)
        #             print('')

        # Now operate on all requested times from *this* NSLC
        # Create empty trace if NSLC returned no data
        if create_empty_trace:
            if not stmp2:
                print('Creating empty trace.')
                stmp = createEmptyTrace(nslc, dt1, dt2, sampling_rate=100)

        #         print('  - Merging stream after we got it all')
        st += stmp
        st = st.merge(method=1, fill_value=fill_value)
        #         print(st)
        #         print('  - Slicing stream')
        # st = st.slice(dt1, dt2, nearest_sample=False) # Ensures that an extra sample is not included at the end (Or use trim?)
        st = st.slice(tstart, tend, nearest_sample=False)  # Slice to entire request time, right?
    #         print(st)
    #         print('')

    # Now operate on all requested times from all requested NSLCs
    #     print('  - Sorting stream by NSLC')
    print('')
    print('  - All the data are downloaded')
    print('  - Sorting stream by NSLC')
    st = sortStreamByNSLClist(st, nslc_list)
    #     print(st)
    #     print('')

    return st


def createEmptyTrace(nslc, t1, t2, sampling_rate=100):
    from obspy import Trace
    from waveformutils.nslcutils import str2nslc

    net, sta, loc, cha = str2nslc(nslc)

    stmp = Trace()
    stmp.stats['network'] = net
    stmp.stats['station'] = sta
    stmp.stats['location'] = loc
    stmp.stats['channel'] = cha
    stmp.stats['sampling_rate'] = sampling_rate
    stmp.stats['starttime'] = t1
    stmp.data = np.zeros(int((t2 - t1) * stmp.stats['sampling_rate']), dtype='int32')

    return stmp

# TO DO
# [ ] get_waveforms_from_files() --> Add option to createEmptyTrace
