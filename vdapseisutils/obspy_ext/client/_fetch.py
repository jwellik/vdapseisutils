"""Waveform download helpers shared by client facades (chunking, dtypes, empty traces)."""


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
def _safe_merge(st, fill_value):
    """
    Merge Traces with same ID, modifying data types if necessary. Modified from code by
    Aaron Wech.

    Args:
        st (:class:`~obspy.swarmmpl.stream.Stream`): Input Stream (modified in-place!)
        fill_value (int, float, str, or None): Passed on to
            :meth:`obspy.swarmmpl.st.Stream.merge`
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
                              verbose=False,
                              **client_kwargs):
    """
    TODO The print status message prints t1 and t2 as what they should be, not necessarily what's actually returned
        (Winston, and maybe other datasources, sometimes return +/-1 sample on either end of the request)

    :param client:
    :param nslc_list:
    :param t1:
    :param t2:
    :param max_download:
    :param fill_value:
    :param create_empty_trace:
    :param empty_samp_rate:
    :param verbose:
    :return:
    """

    from numpy import dtype
    from obspy import UTCDateTime, Stream
    from vdapseisutils.obspy_ext import VStreamID
    from vdapseisutils.utils.obspyutils.streamutils import createEmptyTrace
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

        net, sta, loc, cha = VStreamID(nslc).parts()  # separate string into multiple arguments

        st_nslc = Stream()

        # Create smaller time chunks to download, if necessary
        dtstarts, dtends = timeutils.time_range(t1, t2, freq=max_download)
        for dt1, dt2 in zip(dtstarts, dtends):

            try:
                st_nslc_small = client.get_waveforms(
                    net, sta, loc, cha, dt1, dt2, **client_kwargs
                )  # Call ObsPy function

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

                sr = round(st_nslc_small[0].stats.sampling_rate)
                for tr in st_nslc_small:
                    tr.resample(sr)

            except:
                # stmp2 = Stream()
                st_nslc_small = Stream()

            # Now operating on *this* download segment from *this* NSLC
            st_nslc += st_nslc_small

        # Now operate on all requested times from *this* NSLC
        # Create empty trace if NSLC returned no data
        if st_nslc:
            status = "SUCCESS"
        else:
            if create_empty_trace:
                status = "EMPTY"
                st_nslc = createEmptyTrace(nslc, dtstarts[0], dtends[-1], sampling_rate=empty_samp_rate)
            else:
                st_nslc = Stream()
                status = "FAILED"
        st_nslc = Stream(st_nslc)  # For Trace to be Stream (if only 1 Trace returned, most likely in EMPTY case)

        # Create download message for all requests from this NSLC
        if verbose:
            print("\r" + STATUSMSG.format(nslc=nslc, status=status, ntr=len(st_nslc), npts=__true_npts(st_nslc), t1=dtstarts[0], t2=dtends[-1]))
        # Append to list of Streams for all requests from all NSLCs
        st_all += st_nslc
        # if fill_value:
        #     st_all = st_all.merge(method=1, fill_value=fill_value)

    # print('- Sorting st by NSLC')
    # st_all = sortStreamByNSLClist(st_all, nslc_list)

    return st_all


def get_waveforms_bulk_from_client(
    client,
    bulk,
    max_download="1D",
    fill_value=None,
    create_empty_trace=False,
    empty_samp_rate=100,
    verbose=False,
    **client_kwargs,
):
    """
    Download waveforms for many NSLC/time windows using :func:`get_waveforms_from_client`.

    ``bulk`` follows ObsPy's ``get_waveforms_bulk`` row shape:
    ``(network, station, location, channel, starttime, endtime)``. Rows are grouped by
    identical ``(starttime, endtime)`` so each group uses one chunked fetch pass.
    """
    from collections import defaultdict

    from obspy import Stream, UTCDateTime

    # Use numeric window keys; UTCDateTime is not reliably hashable across ObsPy versions.
    groups = defaultdict(list)
    for row in bulk:
        if len(row) < 6:
            raise ValueError(
                "Each bulk row must be (network, station, location, channel, starttime, endtime)"
            )
        net, sta, loc, cha, t1, t2 = row[:6]
        t1 = UTCDateTime(t1)
        t2 = UTCDateTime(t2)
        loc_s = loc if loc is not None else ""
        nslc = f"{net}.{sta}.{loc_s}.{cha}"
        groups[(float(t1), float(t2))].append(nslc)

    out = Stream()
    for (t1f, t2f), nslc_list in groups.items():
        out += get_waveforms_from_client(
            client,
            nslc_list,
            UTCDateTime(t1f),
            UTCDateTime(t2f),
            max_download=max_download,
            fill_value=fill_value,
            create_empty_trace=create_empty_trace,
            empty_samp_rate=empty_samp_rate,
            verbose=verbose,
            **client_kwargs,
        )
    return out
