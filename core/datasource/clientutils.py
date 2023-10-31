def __true_npts(st):
    """__TRUE_NPTS Returns total number of points from all Traces in Stream
    Intended to report back on how many data points were retrieved vs. expected.
    Usually best to use assuming that all Traces could be merged.
    Merging can't help this calculation bc it will always fill values.
    """

    from obspy import Stream
    st = Stream(st)  # For Trace as Stream

    npts = 0
    for tr in st:
        npts += tr.stats.npts
    return npts

# Code stolen from LiamTooney
import pandas as pd
def _safe_merge(st, fill_value):
    """
    Merge Traces with same ID, modifying data types if necessary. Modified from code by
    Aaron Wech.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Input Stream (modified in-place!)
        fill_value (int, float, str, or None): Passed on to
            :meth:`obspy.core.st.Stream.merge`
    """

    import numpy as np

    try:
        st.merge(fill_value=fill_value)
    except Exception:  # ObsPy raises an Exception if data types are not all identical
        for tr in st:
            if tr.data.dtype != np.dtype(np.int32):
                tr.data = tr.data.astype(np.int32, copy=False)
        st.merge(fill_value=fill_value)


def get_waveforms_from_client(client, nslc_list, t1, t2,
                              max_download="1D",
                              fill_value=None,
                              create_empty_trace=False,
                              empty_samp_rate=100,
                              verbose=False
                              ):

    from numpy import dtype
    from obspy import UTCDateTime, Stream
    from vdapseisutils.core.datasource.nslcutils import str2nslc
    from vdapseisutils.core.datasource.streamutils import createEmptyTrace
    from vdapseisutils.utils import timeutils

    STATUSMSG = '- {nslc:15} | {t1} - {t2} | {status:15} ({ntr:2} traces, {npts:7} samples)'

    # Assert tstart, tend as UTCDateTime
    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2)

    # Assert proper NSLC lists
    if type(nslc_list) is str: nslc_list = [nslc_list]  # Assert NSLC as list

    # st = Stream()
    st_all = Stream()

    # Loop through list of NSLCs
    for nslc in nslc_list:

        net, sta, loc, cha = str2nslc(nslc)

        # stmp = Stream()
        st_nslc = Stream()

        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.time_range(t1, t2, freq=max_download)
        # if verbose:
        #     print(">>> " + STATUSMSG.format(nslc=nslc, status="downloading...", ntr="...", npts="...", t1=dtstarts[0], t2=dtends[-1]), end="\r")
        for dt1, dt2 in zip(dtstarts, dtends):

            # if verbose:
            #     print("\r >>> " + STATUSMSG.format(nslc=nslc, t1=dt1, t2=dt2, status="downloading...", ntr="...", npts="..."))

            try:
                # stmp2 = client.get_waveforms(net, sta, loc, cha, dt1, dt2)  # Call ObsPy function
                st_nslc_small = client.get_waveforms(net, sta, loc, cha, dt1, dt2)  # Call ObsPy function

                # Stolen from Aaron Wech, I think
                # Deal w error when sub-traces have different dtypes
                for tr in st_nslc_small:
                    if tr.data.dtype.name != 'int32':
                        # print("dtype != 'int32'")
                        tr.data=tr.data.astype('int32') # force type int32
                    if tr.data.dtype!=dtype('int32'):
                        # print("dtype != dtype('int32')")
                        tr.data=tr.data.astype('int32') # force type int32
                    # deal with rare error when sub-traces have different sample rates

            except:
                # stmp2 = Stream()
                st_nslc_small = Stream()

            # Now operating on *this* download segment from *this* NSLC
            # stmp += stmp2
            st_nslc += st_nslc_small

        # Now operate on all requested times from *this* NSLC
        # Create empty trace if NSLC returned no data
        # if stmp:
        if st_nslc:
            status = "SUCCESS"
            # stmp = stmp.merge(method=1, fill_value=fill_value)  # Why do this here?
        else:
            if create_empty_trace:
                status = "EMPTY"
                # stmp = createEmptyTrace(nslc, dt1, dt2, sampling_rate=empty_samp_rate)
                st_nslc = createEmptyTrace(nslc, dtstarts[0], dtends[-1], sampling_rate=empty_samp_rate)
            else:
                # stmp = Stream()
                st_nslc = Stream()
                status = "FAILED"
        st_nslc = Stream(st_nslc)  # For Trace to be Stream (if only 1 Trace returned, most likely in EMPTY case)

        # Create download message for all requests from this NSLC
        if verbose:
            print("\r" + STATUSMSG.format(nslc=nslc, status=status, ntr=len(st_nslc), npts=__true_npts(st_nslc), t1=dtstarts[0], t2=dtends[-1]))
        # Append to list of Streams for all requests from all NSLCs
        # st_all += stmp
        st_all += st_nslc
        # if fill_value:
        #     st_all = st_all.merge(method=1, fill_value=fill_value)

    # Now operate on all requested times from all requested NSLCs
    if verbose:
        print('- All the data are downloaded')


    # print('- Sorting st by NSLC')
    # st_all = sortStreamByNSLClist(st_all, nslc_list)

    return st_all
