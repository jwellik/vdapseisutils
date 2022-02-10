def get_waveforms_from_client(client, nslc_list, t1, t2,
                              max_download=86400,
                              fill_value=None,
                              create_empty_trace=False,
                              verbose=False
                              ):
    from obspy import UTCDateTime, Stream

    from vdapseisutils.waveformutils.nslcutils import str2nslc
    from vdapseisutils.waveformutils.streamutils import sortStreamByNSLClist, createEmptyTrace
    from vdapseisutils.waveformutils import timeutils

    # Assert tstart, tend as UTCDateTime
    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2) - 1 / 1000  # Avoids extra data sample being downloaded

    # Assert proper NSLC lists
    if type(nslc_list) is str: nslc_list = [nslc_list]  # Assert NSLC as list

    st = Stream()

    # Loop through list of NSLCs
    for nslc in nslc_list:

        print('- Loading {}'.format(nslc))
        net, sta, loc, cha = str2nslc(nslc)

        stmp = Stream()

        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.createTimeChunks(t1, t2, nsec=max_download, verbose=False)
        for dt1, dt2 in zip(dtstarts, dtends):

            if verbose:
                print('  - Downloading   : {} to {}'.format(dt1, dt2))

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
        st = st.slice(t1, t2, nearest_sample=False)  # Slice to entire request time, right?
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
