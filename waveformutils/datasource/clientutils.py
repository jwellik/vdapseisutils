
def get_waveforms_from_client(client, nslc_list, t1, t2,
                              max_download=86400,
                              fill_value=None,
                              create_empty_trace=False,
                              verbose=False
                              ):

    import datetime as dt
    from obspy import UTCDateTime, Stream

    from vdapseisutils.waveformutils.nslcutils import str2nslc
    from vdapseisutils.waveformutils.streamutils import sortStreamByNSLClist, createEmptyTrace
    from vdapseisutils.waveformutils import timeutils

    from numpy import dtype

    import time

    STATUSMSG = '- {nslc:15} : {status:15} {t1} to {t2}'

    # Assert tstart, tend as UTCDateTime
    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2) - 1 / 1000  # Avoids extra data sample being downloaded

    # Assert proper NSLC lists
    if type(nslc_list) is str: nslc_list = [nslc_list]  # Assert NSLC as list

    st = Stream()

    # Loop through list of NSLCs
    for nslc in nslc_list:

        net, sta, loc, cha = str2nslc(nslc)

        stmp = Stream()

        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.createTimeChunks(t1, t2, nsec=max_download, verbose=False)
        print(STATUSMSG.format(nslc=nslc, status="downloading...", t1=dtstarts[0], t2=dtends[-1]), end="\r")
        for dt1, dt2 in zip(dtstarts, dtends):

            if verbose:
                print(STATUSMSG.format(nslc=nslc, t1=dt1, t2=dt2, status="downloading..."), end="\r")

            try:
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

                stmp2 = stmp2.detrend("linear")
                stmp2 = stmp2.detrend("demean")
                # stmp2 = stmp2.taper(max_percentage=0.01)

                # Stolen from Aaron Wech, I think
                for tr in stmp2: # Deal w error when sub-traces have different dtypes
                    if tr.data.dtype.name != 'int32':
                        # print("dtype != 'int32'")
                        tr.data=tr.data.astype('int32') # force type int32
                    if tr.data.dtype!=dtype('int32'):
                        # print("dtype != dtype('int32')")
                        tr.data=tr.data.astype('int32') # force type int32
                #                     # deal with rare error when sub-traces have different sample rates
                #                 print('  - Merging stream')
                stmp2 = stmp2.merge(fill_value=fill_value)

            except:
                stmp2 = Stream()

            # Now operating on *this* download segment from *this* NSLC
            stmp += stmp2
            stmp = stmp.merge(method=1, fill_value=fill_value)  # Is this necessary?

        # Now operate on all requested times from *this* NSLC
        # Create empty trace if NSLC returned no data
        # NSLC_SUCCESS = []  # List of all NSLCs successfully downloaded
        if stmp:
            print(STATUSMSG.format(nslc=nslc, status="SUCCESS", t1=dt1, t2=dt2))
            stmp = stmp.merge(method=1, fill_value=fill_value)
            # NSLC_SUCCESS.appendstmp[0].id
        else:
            if create_empty_trace:
                stmp = createEmptyTrace(nslc, dt1, dt2, sampling_rate=100)
                print(STATUSMSG.format(nslc=nslc, status="EMPTY", t1=dt1, t2=dt2))
            else:
                stmp = Stream()
                print(STATUSMSG.format(nslc=nslc, status="FAILED", t1=dt1, t2=dt2))

        # print("Adding st += stmp, then merging")
        # print(st)
        st += stmp
        # print(st)
        st = st.merge(method=1, fill_value=fill_value)
        # print(st)

    # Now operate on all requested times from all requested NSLCs
    print('- All the data are downloaded')
    # tclip = dt.timedelta(seconds=0.001)
    # st = st.slice(t1, t2, nearest_sample=False)  # Slice to entire request time, right?
    # st = st.merge(method=1, fill_value=fill_value)
    # st = st.slice(t1, t2-tclip)

    # print('- Sorting stream by NSLC')
    # st = sortStreamByNSLClist(st, nslc_list)

    return st
