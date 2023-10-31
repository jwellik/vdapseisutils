""""

Order of operations when reading Streams from files

--> Provide searchdir, regexp
     ---> Walk thru directories, return matching files to all_files
--> Filter by NSLC, t1, t2
     ---> Load all file headers, return matching files to file_list
--> Load (getWaveforms)
     ---> Load full miniseed file, return Stream, matching files to file_list

>>> ds = DataSource("/path/to/dir", regexp="*.mseed")
 ---> ds.all_files = <walk thru directories>
>>> ds.getWaveforms(["VG.TMKS.00.EHZ"], t1, t2)
 ---> ds.file_list = <load all file headers>
 ---> st


>>> ds = DataSource("/path/to/dir", regexp="*.mseed")
  ---> ds.all_files = <walk thru directories>
>>> file_list = ds.get_filelist(["VG.TMKS.00.EHZ", "VG.PSAG.00.EHZ"], t0, t20)
  ---> <load all file headers>
>>> for nslc in nslc list:
      for t in t0:t20:
        st = ds.getWaveforms(nslc, t, t+1)
        <process batch of data>



"""



def get_all_files(searchdir, filepattern='*'):
    """GET_FILELIST_ALL Returns all files under searchdir that match filepattern"""

    # This method might be faster, but doesn't inherently remove subdirectories
    import glob, os, itertools
    # Generate list of files (all files under top directory)
    flist = list(itertools.chain.from_iterable(glob.iglob(os.path.join(
        root, filepattern)) for root, dirs, files in os.walk(searchdir)))

    ## Might be slow, but only returns files with the desired filepattern
    #import os
    #flist = []
    #for root, dirs, files in os.walk(searchdir, topdown=False):
    #    for name in files:
    #        full_filename = os.path.join(root, name)
    #        if filepattern in full_filename:
    #            print(full_filename)
    #            flist.append(full_filename)

    return flist


def get_filelist(searchdir, nslc_list, t1, t2, filepattern='*', verbose=False):
    """GET_FILELIST Returns all files under searchdir that match filepattern, nslc, t1, and t2"""

    # TODO Print verbose output at end of run, Possible save as output
    # TODO Match SCNL with regular expressions

    from obspy import UTCDateTime, read
    from vdapseisutils.waveformutils.nslcutils import getNSLCstr

    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2)

    flist = get_all_files(searchdir, filepattern=filepattern)

    flist_sub = []
    for f in flist:
        yn = " "  # default (file is not part of sublist)
        status = " "

        # Load header only
        try:
            stmp = read(f, headonly=True)
        except:
            stmp = []

        if stmp:
            status = "o"
            # Check if station is contained in the stas list
            if getNSLCstr(stmp[0]) in nslc_list:  # if stmp[0].stats.station in stas:
                status = "x"
                # Check if contains either start or end time
                ststart = stmp[0].stats.starttime
                stend = stmp[-1].stats.endtime
                if (ststart <= t1 <= stend) or (ststart <= t2 <= stend) or (t1 <= stend and ststart <= t2):
                    flist_sub.append(f)
                    yn = "*"
                    status = "*"

        if verbose:
            print(" {}{} {}".format(yn, status, f))

    return flist_sub


def get_waveforms_from_file_sublist(flist_sub, nslc_list, t1, t2, filepattern='*', fill_value=None,
                             create_empty_trace=False, verbose=False):
    from obspy import UTCDateTime, Stream, Trace, read
    from vdapseisutils.waveformutils.nslcutils import getNSLCstr
    from vdapseisutils.waveformutils.streamutils import sortStreamByNSLClist, createEmptyTrace

    STATUSMSG = '- {nslc:15} : {status:15} {t1} to {t2}'

    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2)

    # Fully load data from file
    stmp = Stream()
    for f in flist_sub:
        tmp = read(f, starttime=t1, endtime=t2)
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
    if verbose:
        print("::: GET_WAVEFORMS_FROM_FILE_SUBLIST ::: ")
    # print(stmp)

    #     # Only grab stations/channels that we want and in order - Modelled after Alicia's code in REDPy
    #     # But this already happens?
    #     st = Stream()
    #     nslc_loaded = []
    #     for s in stmp:
    #         nslc_loaded.append(getNSLCstr(s))
    #     print(nslc_loaded)

    #     # Only grab stations in our nslc_list
    # Modify this to remove '*' and '?' so that if BH in BHZ works
    #     for n in range(len(nslc_list)):
    #         for m in range(len(nslc_loaded)):
    #             print('{} == {} ? --> '.format(nslc_list[n], nslc_loaded[m], nslc_list[n] in nslc_loaded[m]))
    #             if nslc_list[n] in nslc_loaded[m]:
    #                 st = st.append(stmp[m])
    #         print('len(st) {} == n {} == m {}?'.format(len(st),n,m))
    #         if len(st) == n:
    #             print('Could not find file for {}'.format(nslc_list[n]))
    #             trtmp = create_empty_trace(nslc_list[m], tstart, tend)

    if create_empty_trace:
        nslc_loaded = []
        for tr in stmp:
            nslc_loaded.append(getNSLCstr(tr))

        emptytr = Trace()
        for nslc in nslc_list:
            if nslc not in nslc_loaded:
                emptytr = createEmptyTrace(nslc, t1, t2, sampling_rate=100)
        stmp += emptytr

    st = sortStreamByNSLClist(stmp, nslc_list)

    return st
